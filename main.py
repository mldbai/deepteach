#
# main.py
# Francois Maillet, 2016-02-25
# Copyright (c) 2016 Datacratic Inc. All rights reserved.
#

from os.path import join

mldb.log("Staring plugin init...")

# load wrapper so that mldb object behaves like pymldb
mldb = mldb_wrapper.wrap(mldb)

# expose route to serve UI
mldb.plugin.serve_static_folder("/static", "static")


###
# Load embedding code for rt predictions
inceptionUrl = "https://s3.amazonaws.com/public-mldb-ai/models/inception_dec_2015.zip"

mldb.put('/v1/functions/fetch', {
    "type": 'fetcher',
    "params": {}
})

mldb.put('/v1/functions/inception', {
    "type": 'tensorflow.graph',
    "params": {
        "modelFileUrl": 'archive+' + inceptionUrl + '#tensorflow_inception_graph.pb',
        "inputs": 'fetch({url})[content] AS "DecodeJpeg/contents"',
        "outputs": "pool_3"
    }
})

filePrefix = "https://s3.amazonaws.com/public-mldb-ai/datasets/dataset-builder/cache"

###
# Reload cached embeddings
def loadCollection(collection, prefix, limit=-1):
    mldb.log(" >> Loading collection '%s', prefix:%s limit:%d" % (collection, prefix, limit))
    mldb.post("/v1/procedures", {
        "type": "import.text",
        "params": {
            "dataFileUrl": join(prefix, "dataset_creator_images_%s.csv.gz" % collection),
            "select": "* EXCLUDING(rowName)",
            "named": "rowName",
            "limit": limit,
            "outputDataset": {"id": collection, "type": "tabular"}
        }
    })

    rez = mldb.post("/v1/procedures", {
        "type": "import.text",
        "params": {
            "dataFileUrl": join(prefix, "dataset_creator_embedding_%s.csv.gz" % collection),
            "outputDataset": {
                    "id": "embedded_images_%s" % collection,
                    "type": "embedding"
                },
            "select": "* EXCLUDING(rowName)",
            "named": "rowName",
            "where": "rowName IN (select rowName() from %s)" % collection,
            "structuredColumnNames": True
        }
    })
    mldb.log(rez)

    # create nearest neighbour function. this will allow us to quickly find similar images
    mldb.put("/v1/functions/nearest_%s" % collection, {
        "type": "embedding.neighbors",
        "params": {
            "dataset": "embedded_images_%s" % collection
        }
    })


# load built-in collections
limit = -1
if "limit" in mldb.plugin.args:
    limit = mldb.plugin.args["limit"]

for collection in ["recipe", "transport", "pets", "realestate"]:
    loadCollection(collection, filePrefix, limit)

# load any extra collections specified at plugin creation
if "extraCollections" in mldb.plugin.args:
    mldb.log(" >> Loading extra collections")
    for coll in mldb.plugin.args["extraCollections"]:
        if "name" not in coll:
            raise Exception("Key 'name' must be specified for extra collection")

        loadCollection(coll["name"], \
                       coll["prefix"] if "prefix" in coll else filePrefix,
                       coll["limit"] if "limit" in coll else limit)



mldb.log("Ready")

