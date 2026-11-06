"""No network/credentials: durable transport failures must retain model identity."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import urllib.error

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from origami_bench.diagram.api_policy import ApiPolicy,ApiCallError
from origami_bench.diagram.cost_book import CostBook,BudgetExceeded
from origami_bench.diagram.model_identity import RecordingAgent


class ApiCallIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name)
        self.book=CostBook(self.out/'costs.db',100);self.addCleanup(self.book.db.close)
        self.api=ApiPolicy.__new__(ApiPolicy)  # Explicit synthetic fixture; never load credentials.
        self.api.spec=dict(model='declared-A',max_output_tokens=8192,upper_cny_per_million=[1,1],parameters={})
        self.api.book=self.book;self.api.out=self.out;self.api.stage='synthetic'
        self.api.url='https://provider.invalid/chat/completions';self.api.key='SYNTHETIC_CREDENTIAL'
        self.wrapper=RecordingAgent(self.api)

    def request(self,response=None,error=None):
        opener=SimpleNamespace(open=lambda *a,**k:io.BytesIO(json.dumps(response).encode()))
        if error:
            def fail(*a,**k):raise error
            opener.open=fail
        with patch('urllib.request.build_opener',return_value=opener):
            return self.wrapper.request_public('synthetic prompt',[])

    def assert_recorded(self,code):
        request=json.loads((self.out/'call-00001-request.json').read_text())
        response=json.loads((self.out/'call-00001-response.json').read_text())
        self.assertEqual(request['call_id'],1)
        self.assertEqual(response['status'],code)
        self.assertEqual(self.book.db.execute('SELECT status FROM calls WHERE id=1').fetchone()[0],code)
        return response

    def test_http524_records_one_missing_name_after_durable_call(self):
        error=urllib.error.HTTPError(self.api.url,524,'timeout',{},io.BytesIO(b'timed out'))
        with self.assertRaises(ApiCallError) as caught:self.request(error=error)
        self.assertEqual(str(caught.exception),'API http_524; immutable call record 1')
        self.assertEqual(caught.exception.api_call_id,1)
        self.assertIsNone(caught.exception.returned_model)
        self.assertEqual(self.wrapper.returned_model_missing,1)
        self.assertEqual(self.wrapper.returned_models,set())
        self.assert_recorded('http_524')

    def test_provider_error_with_model_records_name_and_reraises(self):
        with self.assertRaises(ApiCallError) as caught:
            self.request(dict(model='returned-A',error=dict(message='unavailable')))
        self.assertEqual(caught.exception.returned_model,'returned-A')
        self.assertEqual(self.wrapper.returned_models,{'returned-A'})
        self.assertEqual(self.wrapper.returned_model_missing,0)
        self.assert_recorded('provider_error')

    def test_http_error_body_model_is_retained_and_credentials_redacted(self):
        body=json.dumps(dict(model='returned-A',error='echo '+self.api.key)).encode()
        error=urllib.error.HTTPError(self.api.url,503,'unavailable',{},io.BytesIO(body))
        with self.assertRaises(ApiCallError) as caught:self.request(error=error)
        self.assertEqual(caught.exception.returned_model,'returned-A')
        self.assertEqual(self.wrapper.returned_models,{'returned-A'})
        self.assertEqual(self.wrapper.returned_model_missing,0)
        saved=self.assert_recorded('http_503')
        self.assertNotIn(self.api.key,json.dumps(saved)+str(caught.exception)+repr(vars(caught.exception)))
        self.assertIn('[REDACTED]',saved['response']['error']['message'])

    def test_transport_exception_records_missing_without_exception_text(self):
        with self.assertRaises(ApiCallError) as caught:
            self.request(error=TimeoutError('unsafe diagnostic '+self.api.key))
        self.assertEqual(self.wrapper.returned_model_missing,1)
        saved=self.assert_recorded('transport_error')
        self.assertNotIn(self.api.key,json.dumps(saved)+str(caught.exception)+repr(vars(caught.exception)))

    def test_success_missing_and_present_are_each_recorded_once(self):
        success=dict(choices=[dict(message=dict(content='{"op":"finish"}'))],usage=dict(prompt_tokens=3,completion_tokens=1))
        result=self.request(success)
        self.assertEqual(result['api_call_id'],1)
        self.assertEqual(self.wrapper.returned_model_missing,1)
        result=self.request(dict(success,model='returned-A'))
        self.assertEqual(result['api_call_id'],2)
        self.assertEqual(self.wrapper.returned_model_missing,1)
        self.assertEqual(self.wrapper.returned_models,{'returned-A'})
        self.assertEqual(self.book.db.execute('SELECT count(*) FROM calls').fetchone()[0],2)

    def test_budget_refusal_is_not_an_api_call(self):
        with patch.object(self.book,'reserve',side_effect=BudgetExceeded('synthetic limit')),patch('urllib.request.build_opener') as network:
            with self.assertRaises(BudgetExceeded):self.wrapper.request_public('public',[])
        network.assert_not_called()
        self.assertEqual(self.wrapper.returned_model_missing,0)
        self.assertEqual(list(self.out.glob('call-*')),[])
        self.assertEqual(self.book.db.execute('SELECT count(*) FROM calls').fetchone()[0],0)

    def test_bad_image_is_not_an_api_call_or_reservation(self):
        with patch('urllib.request.build_opener') as network:
            with self.assertRaises(FileNotFoundError):self.wrapper.request_public('public',[self.out/'absent.png'])
        network.assert_not_called()
        self.assertEqual(self.wrapper.returned_model_missing,0)
        self.assertEqual(self.book.db.execute('SELECT count(*) FROM calls').fetchone()[0],0)

    def test_unstructured_local_failure_cannot_impersonate_call_record(self):
        def fail(*args):raise RuntimeError('API http_524; immutable call record 1')
        wrapper=RecordingAgent(SimpleNamespace(request_public=fail))
        with self.assertRaises(RuntimeError):wrapper.request_public('public',[])
        self.assertEqual(wrapper.returned_model_missing,0)

    def test_structured_protocol_rejects_unsafe_status_and_non_call_id(self):
        for call_id in (None,0,-1,True,'1'):
            with self.assertRaises(ValueError):ApiCallError(call_id,'transport_error')
        with self.assertRaises(ValueError):ApiCallError(1,'echo '+self.api.key)
        self.assertIsNone(ApiCallError(1,'provider_error',{'model':'object'}).returned_model)

if __name__=='__main__':unittest.main()
