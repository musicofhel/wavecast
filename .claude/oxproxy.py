#!/usr/bin/env python3
"""Logging/retrying proxy between claude CLI and OpenRouter (wavecast rw-loop).

Forwards everything to https://openrouter.ai/api, streaming SSE through.
Logs per-request metadata (and small excerpts) to /tmp/oxproxy/oxproxy.jsonl.
If a /v1/messages response completes with NO content blocks that carry
visible output (no text/tool_use deltas seen), retries upstream up to
RETRIES times before giving the CLI the last attempt's stream replay.

Durable home: ~/wavecast/.claude/oxproxy.py (logs are throwaway, script is not).
Launch: plain `python3 ~/wavecast/.claude/oxproxy.py` via Bash run_in_background.
"""
import http.server, json, os, socketserver, sys, time, urllib.request, threading

UPSTREAM = "https://openrouter.ai/api"
LOGDIR = "/tmp/oxproxy"
LOG = LOGDIR + "/oxproxy.jsonl"
REQDIR = LOGDIR + "/reqs"
RETRIES = 8
PORT = 8399

log_lock = threading.Lock()

def logj(obj):
    obj["ts"] = time.strftime("%H:%M:%S")
    with log_lock:
        with open(LOG, "a") as f:
            f.write(json.dumps(obj) + "\n")

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length)
        is_messages = "/v1/messages" in self.path and "count_tokens" not in self.path
        if is_messages:
            try:
                d = json.loads(body)
                changed = []
                if isinstance(d.get("thinking"), dict) and d["thinking"].get("type") == "adaptive":
                    d["thinking"] = {"type": "enabled", "budget_tokens": 6000}
                    changed.append("thinking")
                for k in ("context_management", "output_config"):
                    if k in d:
                        d.pop(k); changed.append(k)
                if d.get("max_tokens", 0) > 8192:
                    d["max_tokens"] = 8192; changed.append("max_tokens")
                if changed:
                    body = json.dumps(d).encode()
                    logj({"rewrote": ",".join(changed)})
            except Exception as e:
                logj({"rewrite_err": repr(e)})
        attempts = RETRIES if is_messages else 1
        if is_messages:
            fn = time.strftime("%H%M%S") + f"-{length}.json"
            with open(REQDIR + "/" + fn, "wb") as bf:
                bf.write(body)
        for attempt in range(1, attempts + 1):
            ok, meta = self.forward(body, deliver=(attempt == attempts) or None)
            meta.update({"path": self.path, "attempt": attempt, "req_bytes": length})
            logj(meta)
            if ok or not is_messages:
                if not meta.get("delivered"):
                    # succeeded on a non-final attempt: deliver this one
                    pass
                return
            # empty/failed and attempts remain: retry without delivering
            if attempt < attempts:
                time.sleep(12 + 4 * attempt)  # space retries; provider load-sheds
        # exhausted; last attempt was delivered regardless

    def forward(self, body, deliver):
        """One upstream attempt. deliver=None on final attempt means always
        stream to client. On non-final attempts we buffer; if the result is
        GOOD we deliver it and report ok; if EMPTY we discard and retry."""
        req = urllib.request.Request(UPSTREAM + self.path, data=body, method="POST")
        for h in ("authorization", "content-type", "anthropic-beta", "anthropic-version", "user-agent", "x-app"):
            v = self.headers.get(h)
            if v:
                req.add_header(h, v)
        t0 = time.time()
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            err_body = e.read()[:2000]
            meta = {"status": e.code, "err": err_body.decode("utf-8", "replace")}
            # on final attempt, relay the error to the client
            if deliver:
                self.send_response(e.code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(err_body)))
                self.end_headers()
                self.wfile.write(err_body)
                meta["delivered"] = True
            return False, meta
        except Exception as e:
            meta = {"status": 0, "err": repr(e)}
            if deliver:
                self.send_response(502)
                self.end_headers()
                meta["delivered"] = True
            return False, meta

        chunks = []
        kinds = set()
        text_chars = 0
        stop_reason = None
        while True:
            chunk = resp.read(16384)
            if not chunk:
                break
            chunks.append(chunk)
        dur = round(time.time() - t0, 1)
        raw = b"".join(chunks)
        # inspect SSE or JSON for visible output
        s = raw.decode("utf-8", "replace")
        for line in s.splitlines():
            if line.startswith("data: "):
                try:
                    d = json.loads(line[6:])
                except Exception:
                    continue
                t = d.get("type")
                if t == "content_block_start":
                    kinds.add(d.get("content_block", {}).get("type"))
                elif t == "content_block_delta":
                    dl = d.get("delta", {})
                    if dl.get("type") == "text_delta":
                        text_chars += len(dl.get("text", ""))
                    elif dl.get("type") == "input_json_delta":
                        text_chars += len(dl.get("partial_json", ""))
                elif t == "message_delta":
                    stop_reason = d.get("delta", {}).get("stop_reason") or stop_reason
        if not kinds and s.lstrip().startswith("{"):
            try:
                d = json.loads(s)
                for c in d.get("content", []):
                    kinds.add(c.get("type"))
                    if c.get("type") == "text":
                        text_chars += len(c.get("text", ""))
                    elif c.get("type") == "tool_use":
                        text_chars += len(json.dumps(c.get("input", {})))
                stop_reason = d.get("stop_reason")
            except Exception:
                pass
        visible = ("text" in kinds and text_chars > 0) or ("tool_use" in kinds)
        meta = {"status": resp.status, "secs": dur, "resp_bytes": len(raw),
                "kinds": sorted(k for k in kinds if k), "text_chars": text_chars,
                "stop_reason": stop_reason, "visible": visible,
                "tail": s[-500:] if not visible else ""}
        good = visible
        if deliver or good:
            self.send_response(resp.status)
            ct = resp.headers.get("content-type", "application/json")
            self.send_header("content-type", ct)
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            meta["delivered"] = True
        return good, meta

    def do_GET(self):
        req = urllib.request.Request(UPSTREAM + self.path)
        for h in ("authorization", "anthropic-version", "user-agent"):
            v = self.headers.get(h)
            if v:
                req.add_header(h, v)
        try:
            resp = urllib.request.urlopen(req, timeout=60)
            raw = resp.read()
            self.send_response(resp.status)
            self.send_header("content-type", resp.headers.get("content-type", "application/json"))
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except urllib.error.HTTPError as e:
            raw = e.read()
            self.send_response(e.code)
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

if __name__ == "__main__":
    os.makedirs(REQDIR, exist_ok=True)
    print(f"oxproxy listening on 127.0.0.1:{PORT} -> {UPSTREAM}", flush=True)
    Server(("127.0.0.1", PORT), H).serve_forever()
