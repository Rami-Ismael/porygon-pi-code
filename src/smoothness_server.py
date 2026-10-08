"""Read-only local visual server backed by the matchup database; never runs battles."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

import smoothness_db as store


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,db,run_id,**kwargs):
        self.db,self.run_id=db,run_id
        super().__init__(*args,**kwargs)

    def send_blob(self,body,content_type):
        self.send_response(200)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route=unquote(urlsplit(self.path).path)
        if route in ['/smoothness/data.js','/smoothness/api/data']:
            try:
                with store.open_db(self.db,readonly=True) as con:
                    document=store.snapshot(con,self.run_id)
                # Asset bytes, as well as numeric results, are available from SQLite.
                for team in document['teams']:
                    for pokemon in team['roster']:
                        for key in ['sprite','itemSprite']:
                            if pokemon.get(key):
                                pokemon[key]='/smoothness/api/assets/'+Path(pokemon[key]).name
                document['storage']['liveDatabase']=True
                body=json.dumps(document,ensure_ascii=False,allow_nan=False)
                if route.endswith('.js'):
                    body='window.SMOOTHNESS_DATA = '+body+';\n'
                self.send_blob(body.encode(),'text/javascript; charset=utf-8' if route.endswith('.js') else 'application/json; charset=utf-8')
            except Exception as error:
                self.log_error('database read failed: %s',error)
                self.send_error(503,'Saved experiment unavailable; no battles were started')
            return
        if route.startswith('/smoothness/api/assets/'):
            name=route.removeprefix('/smoothness/api/assets/')
            if '/' in name or not name.endswith('.png'):
                self.send_error(404)
                return
            with store.open_db(self.db,readonly=True) as con:
                row=con.execute("SELECT content FROM smoothness_artifact WHERE run_id=? AND name=? ORDER BY rowid DESC LIMIT 1",(self.run_id,'visual/assets/'+name)).fetchone()
            if row:
                self.send_blob(bytes(row[0]),'image/png')
            else:
                self.send_error(404,'No stored sprite asset')
            return
        super().do_GET()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,default=store.DEFAULT_DB)
    parser.add_argument('--visual-root',type=Path,required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--port',type=int,default=8770)
    args=parser.parse_args()
    with store.open_db(args.db,readonly=True) as con:
        store.snapshot(con,args.run)
    handler=partial(Handler,directory=str(args.visual_root.resolve()),db=args.db,run_id=args.run)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler)
    print(f'Database-backed visual: http://127.0.0.1:{args.port}/smoothness/',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__=='__main__':
    main()
