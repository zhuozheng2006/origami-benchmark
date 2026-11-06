"""Opt-in HTTPS vision API adapter. No retry, no embedded credentials or provider."""
import os
from pathlib import Path
from urllib.parse import urlsplit
from origami_bench.diagram.api_policy import ApiPolicy, CostBook


def describe_identity():
    return dict(requested_model=os.environ['ORIGAMI_MODEL'], endpoint=os.environ['ORIGAMI_API_BASE'],
                parameters={}, output_cap=8192, adapter_version='public-api-v2.1',
                output_parameter=os.environ.get('ORIGAMI_OUTPUT_PARAMETER', 'max_tokens'))


class PublicAPI(ApiPolicy):
    def __init__(self):
        endpoint = os.environ['ORIGAMI_API_BASE'].rstrip('/')
        parsed = urlsplit(endpoint)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('An explicit HTTPS endpoint without URL credentials is required')
        self.url = endpoint+'/chat/completions'
        self.key = os.environ['ORIGAMI_API_KEY']
        self.out = Path(os.environ['ORIGAMI_API_LOGDIR'])
        self.out.mkdir(parents=True, exist_ok=False)
        self.book = CostBook(os.environ['ORIGAMI_COST_LEDGER'], limit=float(os.environ['ORIGAMI_COST_LIMIT']))
        rates = [float(os.environ['ORIGAMI_INPUT_RATE']), float(os.environ['ORIGAMI_OUTPUT_RATE'])]
        if min(rates) < 0:
            raise ValueError('Rates must be nonnegative upper estimates per million tokens')
        self.spec = dict(model=os.environ['ORIGAMI_MODEL'], max_output_tokens=8192,
                         upper_cny_per_million=rates, parameters={})
        output_parameter=os.environ.get('ORIGAMI_OUTPUT_PARAMETER','max_tokens')
        if output_parameter not in ('max_tokens','max_completion_tokens'):
            raise ValueError('Unsupported output token parameter')
        self.spec['output_parameter']=output_parameter
        self.stage = 'public_release_evaluation'
        self.metadata = describe_identity()

    def close(self):
        self.book.db.close()


def create():
    return PublicAPI()
