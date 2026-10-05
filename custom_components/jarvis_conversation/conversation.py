"""Home Assistant owns exposure, validation, execution and conversation state."""
import asyncio
import logging
import json
import re

from probatio import to_openapi
import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.components.homeassistant.llm import async_get_exposed_entities
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import dt as dt_util

from .const import DOMAIN, PROMPT
from .planner import Planner, result_speech, select_tools

LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass, entry, async_add_entities):
    async_add_entities([JarvisConversation(entry)])


class JarvisConversation(conversation.ConversationEntity):
    _attr_name = 'Jarvis'
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL
    _attr_supported_languages = ['en']
    _attr_should_poll = False

    def __init__(self, entry):
        self.entry = entry
        self._attr_unique_id = entry.entry_id

    @property
    def supported_languages(self):
        return ['en']

    async def _async_handle_message(self, user_input, chat_log):
        settings = {**self.entry.data, **self.entry.options}
        planner = Planner(async_get_clientsession(self.hass), settings)
        # Same deterministic clock shortcut as the repository's microphone client.
        if re.fullmatch(r"(?:what(?:'s| is)|tell me) (?:the )?(?:time|date|day)(?: is it)?[?.! ]*", user_input.text.strip(), re.I) or re.fullmatch(r'what (?:time|day|date) is it[?.! ]*', user_input.text.strip(), re.I):
            now = dt_util.now()
            reply = now.strftime('It is %I:%M %p.') if 'time' in user_input.text.lower() else now.strftime('Today is %A, %B %d, %Y.')
            chat_log.async_add_assistant_content_without_tools(conversation.AssistantContent(agent_id=user_input.agent_id, content=reply))
            LOGGER.info('Jarvis planner stage=clock calls=0')
            return conversation.async_get_result_from_chat_log(user_input, chat_log)
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN), llm.LLM_API_ASSIST,
                PROMPT, user_input.extra_system_prompt)
        except conversation.ConverseError as err:
            return err.as_conversation_result()
        api = chat_log.llm_api
        tools = {tool.name: tool for tool in api.tools}
        schemas = [{'name': tool.name, 'description': tool.description or tool.name,
                    'parameters': to_openapi(tool.parameters, custom_serializer=api.custom_serializer)}
                   for tool in api.tools]
        schemas = select_tools(user_input.text, schemas)

        def validate_args(name, args):
            try:
                return tools[name].parameters(args)
            except vol.Invalid as exc:
                raise ValueError('Invalid tool arguments') from exc

        history = [{'role': content.role, 'content': content.content[:500]}
                   for content in chat_log.content[1:-1]
                   if content.role in ('user', 'assistant') and content.content][-4:]
        # A full-home prompt can exceed Groq's token quota. Send a ranked subset,
        # containing only HA-exposed names/areas, never an unfiltered state dump.
        words = set(re.findall(r'[a-z0-9]+', user_input.text.lower()))
        exposed = async_get_exposed_entities(self.hass, 'conversation', include_state=False)
        ranked = sorted(exposed.values(), key=lambda item: len(words & set(re.findall(r'[a-z0-9]+', json.dumps(item).lower()))), reverse=True)
        catalog = []
        for item in ranked[:24]:
            if len(json.dumps(catalog + [item])) > 4000:
                break
            catalog.append(item)
        context = (f'Current local date/time: {dt_util.now().isoformat()}. '
            'Use homeassistant__GetLiveContext for current device values. '
            'This is a subset of exposed entities. Ask for clarification if a target is missing or ambiguous. '
            'Conditional device changes are unsupported; ask the user for an explicit action. '
            'Exposed entities: ' + json.dumps(catalog, ensure_ascii=False))
        try:
            async with asyncio.timeout(65):
                plan = await planner.plan(user_input.text, schemas, context, history, validate_args)
            LOGGER.info('Jarvis planner stage=%s calls=%d', plan.stage, len(plan.calls))
            replies = []
            # One call at a time, so a failure cannot trigger later actions.
            # Planning/fallback has ended: never replan or replay an executed call.
            for call in plan.calls:
                tool_input = llm.ToolInput(tool_name=call['name'], tool_args=call['arguments'])
                failed = True
                received_result = False
                async with asyncio.timeout(30):
                    async for result in chat_log.async_add_assistant_content(
                        conversation.AssistantContent(agent_id=user_input.agent_id, tool_calls=[tool_input])):
                        speech, failed = result_speech(result.tool_result)
                        replies.append(speech)
                        received_result = True
                if not received_result:
                    replies.append('I could not confirm that request. Please check the device.')
                if failed:
                    break
            reply = ' '.join(replies)[:900] if replies else plan.reply
        except TimeoutError:
            reply = 'The request timed out. I have not retried any device action; please check its state.'
        except Exception as exc:
            # Exception bodies can contain service URLs, credentials or device data.
            LOGGER.warning('Jarvis conversation failed: %s', type(exc).__name__)
            reply = 'I could not complete that request. Please check the device before trying again.'
        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(agent_id=user_input.agent_id, content=reply))
        return conversation.async_get_result_from_chat_log(user_input, chat_log)
