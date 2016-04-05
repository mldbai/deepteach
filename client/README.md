The client code use node and npm.
Below are instructions to install on ubuntu.

> sudo apt-get install node npm

Ubuntu's default program node is Amateur Packet Radio Node.
Below code make sure node point to node.js

> sudo ln -s /usr/bin/nodejs /usr/bin/node

> export PATH=/usr/bin:./node_modules/.bin/:$PATH

> cd client

this will install all js lib and the following executables: tsc, webpack, typings into ./node_modules/.bin
> npm install

rm typings folder so typings can generate correct typings/tsd.d.ts
> rm -rf typings

this will download all .d.ts into typings/ and create typings/tsd.d.ts
> typings install

this will create build/bundle.js and it's .map
> webpack


Run below code in mldb's ipython interface to init the plugin.
after run mldb with command like below:

docker run --name=mldb --rm=true -v /home/hao/mldb_data:/mldb_data -e MLDB_IDS="`id`" -p 127.0.0.1:8000:80 quay.io/datacratic/mldb:v2016.03.22.0

goto url with browser
http://localhost:8000/ipy/tree

click New/Python 2

Enter below text in the code block:

from pymldb import Connection
mldb = Connection("http://localhost")
mldb.put('/v1/plugins/dataset-builder', {"type": "python","params": {"address": "file:///mldb_data/dataset-builder"}})

Click run cell button (the one with a solid black triangle point to the right and a bar)
wait the 201 Created response displayed on screen.

Now goes to
http://localhost:8000/v1/plugins/dataset-builder/routes/static/index.html
