"""Allowlisted experiment identity. No credentials or arbitrary adapter metadata."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from urllib.parse import urlsplit

VERSION = 'model-identity-v1'
PARAMETERS = {'temperature', 'top_p', 'seed', 'reasoning_effort', 'presence_penalty',
              'frequency_penalty', 'stop', 'thinking', 'response_format'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def safe_parameters(parameters):
    if not isinstance(parameters, dict):
        raise ValueError('Model parameters must be a dictionary')
    unknown = set(parameters) - PARAMETERS
    if unknown:
        raise ValueError('Unsupported parameter keys; explicitly extend the identity schema')
    result = {}
    for key, value in parameters.items():
        if key in ('thinking', 'response_format'):
            allowed = {'type', 'budget_tokens'} if key == 'thinking' else {'type'}
            if not isinstance(value, dict) or set(value) - allowed:
                raise ValueError('Unsupported nested model parameters')
        result[key] = value
    # Reject non-JSON / nonfinite input instead of producing unstable identities.
    digest(result)
    return result


def normalize_identity(metadata, adapter):
    if not isinstance(metadata, dict):
        raise ValueError('Model identity metadata is required')
    model = metadata.get('requested_model', metadata.get('model'))
    provider = metadata.get('provider', metadata.get('endpoint'))
    if not isinstance(model, str) or not model.strip() or not isinstance(provider, str) or not provider.strip():
        raise ValueError('Requested model and provider identity are required')
    if '://' in provider:
        url = urlsplit(provider)
        if not url.hostname or url.username or url.password:
            raise ValueError('Provider identity must not contain URL credentials')
        provider = url.scheme + '://' + url.netloc  # strip query, path and fragment
    cap = metadata.get('output_cap', metadata.get('max_output_tokens'))
    if type(cap) is not int or cap != 8192:
        raise ValueError('This release requires a declared 8192-token output cap')
    module_name = adapter.split(':')[0]
    module = importlib.import_module(module_name)
    source = getattr(module, '__file__', None)
    if source is None or not Path(source).is_file():
        raise ValueError('Adapter source is required for reproducible identity')
    value = dict(schema=VERSION, requested_model=model.strip(), provider=provider.strip(),
                 parameters=safe_parameters(metadata.get('parameters', {})), output_cap=cap,
                 output_parameter=metadata.get('output_parameter', 'max_tokens'),
                 adapter=adapter, adapter_version=str(metadata.get('adapter_version', 'source-pinned')),
                 adapter_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    if value['output_parameter'] not in ('max_tokens', 'max_completion_tokens'):
        raise ValueError('Unknown output cap parameter')
    value['identity_sha256'] = digest(value)
    return value


def validate_identity(value):
    if not isinstance(value, dict) or value.get('schema') != VERSION:
        return False
    expected = {'schema','requested_model','provider','parameters','output_cap','output_parameter',
                'adapter','adapter_version','adapter_sha256','identity_sha256'}
    if set(value) != expected or not all(value.get(k) for k in ('requested_model','provider','adapter','adapter_sha256')):
        return False
    try:
        safe_parameters(value['parameters'])
        return value['output_cap'] == 8192 and value['identity_sha256'] == digest({k:v for k,v in value.items() if k != 'identity_sha256'})
    except (ValueError, TypeError):
        return False


def declared_identity(adapter, identity_file=None):
    """No factory construction or request: declaration must be side-effect free."""
    if identity_file:
        return normalize_identity(json.loads(Path(identity_file).read_text()), adapter)
    module = importlib.import_module(adapter.split(':')[0])
    describe = getattr(module, 'describe_identity', None)
    if not callable(describe):
        raise ValueError('Supply --identity JSON or adapter describe_identity() before evaluation')
    return normalize_identity(describe(), adapter)


class RecordingAgent:
    def __init__(self, agent):
        self.agent = agent
        self.returned_models = set()
        self.returned_model_missing = 0
    def request_public(self, text, images):
        result = self.agent.request_public(text, images)
        returned = result.get('returned_model')
        if isinstance(returned, str) and returned.strip():
            self.returned_models.add(returned)
        else:
            self.returned_model_missing += 1
        return result
    def close(self):
        if hasattr(self.agent, 'close'):
            self.agent.close()


def task_policy(spec, task, data):
    if spec['episode'] == 'step':
        from .single_panel_policy import select_policy
        return select_policy(task, Path(data)/'public'/task['image'])
    if spec['episode'] in ('document', 'hard'):
        from . import document_policy
        return document_policy
    from . import layer_policy_v15
    return layer_policy_v15


def policy_digest(policy):
    return hashlib.sha256(Path(policy.__file__).read_bytes()).hexdigest()
