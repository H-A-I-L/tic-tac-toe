from socketserver import ThreadingMixIn
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, SimpleHTTPRequestHandler
import os
import sys
import json
import logging

# The logger is used to log the results to a file in a thread safe environment.
# Use the log() function to log anything
# When wrting to file is needed use the `print_to_file` parameter
LOGGER = logging.getLogger("Server")
FILE_OUT_LOGGER = logging.getLogger("File")
streamformatter = logging.Formatter(fmt= "\033[33m%(asctime)s:\033[32m%(name)s:\033[34;1m%(levelname)s \033[0m%(message)s")
fileformatter = logging.Formatter(fmt= "%(message)s")
streamHandler = logging.StreamHandler(sys.stdout)
streamHandler.setFormatter(streamformatter)
streamHandler.setLevel(logging.INFO)
fileHandler = logging.FileHandler("results.out")
fileHandler.setFormatter(fileformatter)
fileHandler.setLevel(logging.INFO)
LOGGER.handlers = []
FILE_OUT_LOGGER.handlers = []
LOGGER.addHandler(streamHandler)
FILE_OUT_LOGGER.addHandler(fileHandler)
LOGGER.setLevel(logging.DEBUG)
FILE_OUT_LOGGER.setLevel(logging.DEBUG)

def log(message, level=logging.INFO, print_to_file = False):
    LOGGER.log(level, message)
    if print_to_file:
        FILE_OUT_LOGGER.log(level, message)

class Handler(SimpleHTTPRequestHandler):
    
    def do_GET(self):
        # This section was copied from the SimpleHTTPRequestHandler source of cpython
        f = self.send_head()
        if f:
            try:
                self.copyfile(f, self.wfile)
            finally:
                f.close()
        else:
            # handle other requests here 
            self.send_response(200)
            self.end_headers()
            message =  threading.currentThread().getName()
            self.wfile.write(message.encode('utf-8'))
            self.wfile.write('\n'.encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = json.loads(self.rfile.read(content_length))
        log("Recieved data:")
        log(",".join([str(i) for i in post_data['grid']+[post_data['winner']]]), print_to_file=True)
        self.send_response(200)
        self.end_headers()

    # This was copied from the SimpleHTTPRequestHandler source code of cpython
    def send_head(self):
        """Common code for GET and HEAD commands.
        This sends the response code and MIME headers.
        Return value is either a file object (which has to be copied
        to the outputfile by the caller unless the command was HEAD,
        and must be closed by the caller under all circumstances), or
        None, in which case the caller has nothing further to do.
        """
        path = self.translate_path(self.path)
        f = None
        if os.path.isdir(path):
            if not self.path.endswith('/'):
                # redirect browser - doing basically what apache does
                self.send_response(301)
                self.send_header("Location", self.path + "/")
                self.end_headers()
                return None
            for index in "index.html", "index.htm":
                index = os.path.join(path, index)
                if os.path.exists(index):
                    path = index
                    break
            else:
                return self.list_directory(path)
        ctype = self.guess_type(path)
        try:
            f = open(path, 'rb')
        except IOError:
            #instead of raising an exception, return None
            return None
        try:
            self.send_response(200)
            self.send_header("Content-type", ctype)
            fs = os.fstat(f.fileno())
            self.send_header("Content-Length", str(fs[6]))
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()
            return f
        except:
            f.close()
            raise


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

if __name__ == '__main__':
    server = ThreadedHTTPServer(('localhost', 8080), Handler)
    log('Starting server, use <Ctrl-C> to stop')
    log('Server running on: localhost:8080')
    server.serve_forever()
    
