import io
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from origami_bench.diagram import public_remote_backend as bridge


class PublicRemoteBackendTests(unittest.TestCase):
    def env(self):
        return dict(ORIGAMI_BACKEND_MODE='ssh',ORIGAMI_SSH_HOST='4090B',
                    ORIGAMI_REMOTE_ROOT='/data/project',ORIGAMI_REMOTE_PYTHON='/data/env/bin/python',ORIGAMI_CPU_GROUP='5')

    def test_explicit_remote_configuration_and_cpu_groups(self):
        e=self.env(); self.assertEqual(bridge.settings_from_environment(e)['cpu_group'],5)
        for group in ['0','3','7']:
            with self.assertRaises(ValueError):bridge.settings_from_environment({**e,'ORIGAMI_CPU_GROUP':group})
        for host in ['-oProxyCommand=bad','4090B;bad','']:
            with self.assertRaises(ValueError):bridge.settings_from_environment({**e,'ORIGAMI_SSH_HOST':host})
        with self.assertRaises(ValueError):bridge.settings_from_environment({**e,'ORIGAMI_REMOTE_ROOT':'/data/../private'})

    def test_command_quotes_paths_preserves_proxy_and_masks_gpu_child(self):
        cmd=bridge.worker_command('/data/my project','/data/env python',4,'layer_worker.py')
        self.assertIn("cd '/data/my project'",cmd)
        parsed=shlex.split(cmd.split(' && exec ',1)[1])
        self.assertEqual(parsed[:4],['flock','-n','/tmp/origami-public-cpu-group-4.lock','env'])
        self.assertIn('CUDA_VISIBLE_DEVICES=',parsed)
        self.assertNotIn('unset',parsed)
        self.assertEqual(parsed.count('/data/env python'),2)

    def test_factory_local_default_and_explicit_remote(self):
        with patch.dict(os.environ,{},clear=True),patch.object(bridge,'LocalLayerClient',return_value='local') as local:
            self.assertEqual(bridge.configured_backend('horizontal'),'local'); local.assert_called_once()
        with patch.dict(os.environ,self.env(),clear=True),patch.object(bridge,'SSHLayerClient',return_value='remote') as remote:
            self.assertEqual(bridge.configured_backend('horizontal'),'remote')
            self.assertEqual(remote.call_args.kwargs['cpu_group'],5)

    def test_ssh_children_do_not_inherit_credentials_or_ledger(self):
        env={'ORIGAMI_API_KEY':'secret','ORIGAMI_COST_LEDGER':'private.sqlite',
             'ORIGAMI_MODEL':'model','HTTPS_PROXY':'http://127.0.0.1:17897','SSH_AUTH_SOCK':'agent'}
        safe=bridge.transport_environment(env)
        self.assertEqual(safe,{'HTTPS_PROXY':'http://127.0.0.1:17897','SSH_AUTH_SOCK':'agent'})

    def test_source_manifest_never_contains_credentials_or_ledger(self):
        manifest=bridge.source_manifest('horizontal','layer_worker.py')
        self.assertIn('scripts/layer_worker.py',manifest)
        self.assertTrue(any(p.startswith('tasks/') for p in manifest))
        self.assertTrue(all('.local' not in p and 'cost' not in p and 'api-' not in p for p in manifest))
        with self.assertRaises(ValueError):bridge.source_manifest('../secret','layer_worker.py')

    def test_source_mismatch_stops_before_worker_launch(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(bridge,'ROOT',Path(tmp)),patch.object(bridge,'source_manifest',return_value={'worker':'hash'}),patch.object(bridge.subprocess,'run',return_value=SimpleNamespace(returncode=2,stdout='mismatch')),patch.object(bridge.subprocess,'Popen') as popen:
            with self.assertRaises(RuntimeError):bridge.SSHLayerClient('horizontal',host='4090B',remote_root='/data/project',python='/data/env/python')
            popen.assert_not_called()

    def test_mock_worker_launch_and_close_no_network(self):
        process=MagicMock(); process.poll.return_value=None
        state={'hash':'initial'}
        replies=[dict(material=[[0,0,0]],configuration={},state=state),{'ok':True}]
        with tempfile.TemporaryDirectory() as tmp,patch.object(bridge,'ROOT',Path(tmp)),patch.object(bridge,'source_manifest',return_value={'worker':'hash'}),patch.object(bridge.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps({'ok':True,'checked':1}))),patch.object(bridge.subprocess,'Popen',return_value=process) as popen,patch.object(bridge.selectors,'DefaultSelector'),patch.object(bridge.SSHLayerClient,'_request',side_effect=replies):
            client=bridge.SSHLayerClient('horizontal',host='4090B',remote_root='/data/project',python='/data/env/python',cpu_group=6)
            self.assertEqual(client.configuration['cpu_group'],6)
            self.assertEqual(client.configuration['inference_location'],'local')
            self.assertIn('--group 6',popen.call_args.args[0][-1])
            client.close();client.close()

    def test_artifact_path_does_not_escape_own_worker_root(self):
        client=object.__new__(bridge.SSHLayerClient);client.remote_root='/data/project'
        self.assertEqual(client._artifact_name('/data/project/outputs/interactive-codim/'+'a'*32),'a'*32)
        for p in ['/data/project/outputs/interactive-codim/../secret','/data/other/outputs/interactive-codim/'+'a'*32,'/data/project/outputs/interactive-codim/abc']:
            with self.assertRaises(ValueError):client._artifact_name(p)

    def test_physics_signature_mismatch_closes_worker_before_agent_use(self):
        process=MagicMock();process.poll.return_value=None
        replies=[dict(material=[],configuration={},state={'hash':'initial'}),{'ok':True}]
        with tempfile.TemporaryDirectory() as tmp:
            expected=Path(tmp)/'physics.json';expected.write_text(json.dumps({'native':'expected'}))
            with patch.dict(os.environ,{'ORIGAMI_PHYSICS_SIGNATURE_FILE':str(expected)}),patch.object(bridge,'ROOT',Path(tmp)),patch.object(bridge,'source_manifest',return_value={'worker':'hash'}),patch.object(bridge.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps({'ok':True,'checked':1}))),patch.object(bridge.subprocess,'Popen',return_value=process),patch.object(bridge.selectors,'DefaultSelector'),patch.object(bridge.SSHLayerClient,'_request',side_effect=replies) as request,patch.object(bridge,'signature',return_value={'native':'changed'}):
                with self.assertRaisesRegex(RuntimeError,'physics signature differs'):
                    bridge.SSHLayerClient('horizontal',host='4090B',remote_root='/data/project',python='/data/env/python')
                self.assertEqual(request.call_args.args[0],{'op':'close'})


if __name__=='__main__':unittest.main()
