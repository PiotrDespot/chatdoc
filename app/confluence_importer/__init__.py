# __init__.py

from .exporter import Exporter
from .converter import Converter


class ConfluenceImporter:
    def __init__(self, params):
        self.params = params

    def run(self):
        url = self.params.get("url", "")
        username = self.params.get("username", "")
        token = self.params.get("token", "")
        out_dir = self.params.get("out_dir", "")
        space = self.params.get("space", None)
        no_attach = self.params.get("no_attach", False)
        no_fetch = self.params.get("no_fetch", False)

        if not no_fetch:
            dumper = Exporter(
                url=url,
                username=username,
                token=token,
                out_dir=out_dir,
                space=space,
                no_attach=no_attach
            )
            dumper.dump()

        converter = Converter(out_dir=out_dir)
        converter.convert()
