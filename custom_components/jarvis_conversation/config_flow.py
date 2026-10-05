"""UI configuration. Keys are stored in HA's private config entry."""
import asyncio
import aiohttp
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import TextSelector, TextSelectorConfig, TextSelectorType

from .const import DOMAIN, DEFAULT_MODEL
from .planner import rpc_url


def schema(settings):
    password = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))
    return vol.Schema({
        vol.Required('backend_url', default=settings.get('backend_url', 'ws://inference.example.invalid:8765')): str,
        vol.Required('backend_token', default=settings.get('backend_token', '')): password,
        vol.Optional('groq_api_key', default=settings.get('groq_api_key', '')): password,
        vol.Required('groq_model', default=settings.get('groq_model', DEFAULT_MODEL)): str,
        vol.Required('min_confidence', default=settings.get('min_confidence', 0.8)): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
    })


async def validate(hass, data):
    url = rpc_url(data['backend_url'])
    if len(data['backend_token']) < 32:
        raise ValueError('Backend token too short')
    async with asyncio.timeout(10):
        async with async_get_clientsession(hass).ws_connect(url,
            headers={'Authorization': 'Bearer ' + data['backend_token']}, max_msg_size=128000) as ws:
            ready = await ws.receive_json()
            if ready.get('type') != 'ready' or ready.get('protocol') != 2:
                raise ValueError('Incompatible worker')


class JarvisConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input=None):
        errors = {}
        if user_input is not None:
            try:
                await validate(self.hass, user_input)
            except (aiohttp.ClientError, TimeoutError, ValueError):
                errors['base'] = 'cannot_connect'
            else:
                await self.async_set_unique_id(rpc_url(user_input['backend_url']))
                self._abort_if_unique_id_configured()
                return self.async_create_entry(title='Jarvis Conversation', data=user_input)
        return self.async_show_form(step_id='user', data_schema=schema(user_input or {}), errors=errors)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return JarvisOptionsFlow()


class JarvisOptionsFlow(config_entries.OptionsFlow):
    async def async_step_init(self, user_input=None):
        errors = {}
        if user_input is not None:
            try:
                await validate(self.hass, user_input)
            except (aiohttp.ClientError, TimeoutError, ValueError):
                errors['base'] = 'cannot_connect'
            else:
                return self.async_create_entry(title='', data=user_input)
        settings = user_input or {**self.config_entry.data, **self.config_entry.options}
        return self.async_show_form(step_id='init', data_schema=schema(settings), errors=errors)
