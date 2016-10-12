#
# routes.py
# Francois Maillet, 2016-02-25
# Copyright (c) 2016 Datacratic Inc. All rights reserved.
#

import json, os, datetime, math
from operator import itemgetter
import functools, urllib, time
import binascii

mldb2 = mldb_wrapper.wrap(mldb)

EMBEDDING_DATASET = "embedded_images"


def preProcessData():
    rp = json.loads(mldb.plugin.rest_params.payload)

    mldb.log(rp)

    inputIndex = -1
    deploy = False
    dataset = None
    numBags = 50
    numRndFeats = 0.1

    if "deploy" in rp:
        deploy = rp["deploy"]
    
    if "prefix" in rp:
        prefix = rp["prefix"]
    if "numBags" in rp:
        numBags = int(rp["numBags"])
    if "numRndFeats" in rp:
        numRndFeats = float(rp["numRndFeats"])

    if "dataset" not in rp:
        return ("Dataset needs to be specified", 400)
    dataset = rp["dataset"]

    if "input" not in rp:
        mldb.log(rp)
        return ("Invalid input! (1)", 400)

    data = json.loads(rp["input"])
    if "a" not in data or "b" not in data:
        return ("Data dict must contain keys a and b", 400)


    groups = [set(data["a"]), set(data["b"]), set(data["ignore"])]
    for idx, name in enumerate(("a", "b")):
        if len(groups[idx]) == 0:
            return ("Data group '%s' cannot be empty!" % name, 400)

    return data, groups, deploy, dataset, prefix, numBags, numRndFeats



def getPrediction():
    input_data ={}

    for elem in mldb.plugin.rest_params.payload.split("&"):
        k,v = elem.split("=")
        input_data[k] = urllib.unquote(v)

    if "deploy_id" not in input_data or "image_url" not in input_data:
        return ("deploy_id and image_url must be passed to endpoint!", 400)

    mldb.log(input_data)

    import urllib2
    try:
        response = urllib2.urlopen(input_data["image_url"])
        if(response.getcode() != 200):
            return ("Error opening image: %s. URL: %s" % (str(response.info()), input_data["image_url"]), 400)
    except Exception as e:
        return ("Error opening image: %s. URL: %s" % (str(e), input_data["image_url"]), 400)

    unableToGetMimeType = False
    if 'Content-type' not in response.info():
        unableToGetMimeType = True
        mldb.log(str(response.info()))
        #return ("Real-time prediction only supports JPEG images. Unable to determine mine type", 400)
    else:
        mime = response.info()['Content-type']
        mldb.log(str(mime))
        if not mime.endswith("jpeg"):
            return ("Real-time prediction only supports JPEG images. Mime type was '%s'" % mime, 400)

    try:
        score_query_rez = mldb2.query("""
                SELECT explorator_cls_%s(
                                {
                                    features: {*}
                                }) as scores
                FROM (
                    SELECT inception({url: '%s'}) as *
                )
            """ % (input_data["deploy_id"], input_data["image_url"]))
    except Exception as e:
        if unableToGetMimeType:
            return ("Error when trying to score image. Could not determine mime type. Probably not a JPEG image", 400)

        return ("Error scoring image: %s. URL: %s" % (str(e), input_data["image_url"]), 400)



    mldb.log(score_query_rez)
    score = score_query_rez[1][1]

    dataset = mldb2.query("""
            select score.score from predictions_%s
            where training_labels.label = 0 and training_labels.weight = 1
            order by score.score ASC LIMIT 1
        """ % input_data["deploy_id"])
    min_score_for_pos = dataset[1][1]

    dataset = mldb2.query("""
            select score.score from predictions_%s
            order by score.score DESC LIMIT 1
        """ % input_data["deploy_id"])
    maxScore = dataset[1][1]


    rez = mldb2.query("""
        SELECT * FROM merge(
            (
                select 'A' as class, datasetName, imagePrefix
                from predictions_%s
                where training_labels.label = 0 and training_labels.weight = 1
                order by score.score DESC LIMIT 5
            ),
            (
                select 'B' as class, datasetName, imagePrefix
                from predictions_%s
                where training_labels.label = 1
                order by score.score ASC LIMIT 5
            )
        )
    """ % (input_data["deploy_id"], input_data["deploy_id"]))

    example_images = {"A": [], "B": []}
    for elem in rez[1:]:
        example_images[elem[1]].append([elem[0], elem[2], elem[3]])


    return_val = {
            "example_image": example_images,
            "similarity": {
                "score": score,
                "threshold": min_score_for_pos,
                "max": maxScore
            },
            "prediction": "A" if score >= min_score_for_pos * 0.9 else "B"
        }


    return (return_val, 200)





def getSimilar(cls_func_name="explorator_cls"):

    data, groups, doDeploy, datasetName, prefix, numBags, numRndFeats = preProcessData()

    embeddingDataset = EMBEDDING_DATASET + "_" + datasetName

    run_id = str(binascii.hexlify(os.urandom(16)))
    cls_func_name += "_" + run_id

    # keep track of ressources to delete
    to_delete = []

    dataset_config = {
        'type'    : 'sparse.mutable',
        'id'      : "training_labels_pos_" + run_id if not doDeploy else "training_labels_"+run_id
    }

    to_delete.append("/v1/datasets/" + dataset_config["id"])
    dataset = mldb.create_dataset(dataset_config)
    now = datetime.datetime.now()


    to_add = []
    already_added = set()

    times = {}
    t0 = time.time()
    for lbl, imgs in enumerate(groups):
        for img in imgs:
            if img in already_added: continue
            dataset.record_row(img, [["label", lbl, now], ["weight", 1, now]])
            already_added.add(img)

            # if it's the positive group, look at nearest neighbhours to add extra
            # labels in positive class in case we don't have enough
            if lbl == 0 and not doDeploy:
                # will return list of ["slid","596e1ca6687cd301",1.9799363613128662]
                neighbours = mldb2.query("select nearest_%s({coords: '%s'})[neighbors] as *" % (datasetName, img))

                for nName in neighbours[1][1:]:
                    to_add.append((nName, [["label", lbl, now], ["weight", 0.25, now]]))


    # add the positive nearest neighbours if they were't added as
    # explicit negative examples
    for row, cols in to_add:
        if row in already_added: continue
        dataset.record_row(row, cols)
        already_added.add(row)

    dataset.commit()

    t1 = time.time()
    times["1 - add posex nearest neighbour"] = t1-t0

    t0 = time.time()
    # now add all remaining examples as low weight negative examples
    if not doDeploy:
        to_delete.append("/v1/datasets/training_labels_neg_" + run_id)
        mldb2.post("/v1/procedures", {
            "type": "transform",
            "params": {
                "inputData": """
                    SELECT label: 1, weight: 0.001
                    FROM %s
                    WHERE NOT (rowName() IN (SELECT rowName() FROM training_labels_pos_%s))
                """ % (embeddingDataset, run_id),
                "outputDataset": "training_labels_neg_"+run_id,
            }
        })

        to_delete.append("/v1/datasets/training_labels_" + run_id)
        mldb2.put("/v1/datasets/training_labels_" + run_id, {
            "type": "merged",
            "params": {
                "datasets": [
                    {"id": "training_labels_neg_"+run_id},
                    {"id": "training_labels_pos_"+run_id}
                ]
            }
        })

        t1 = time.time()
        times["2 - add negex low weight"] = t1-t0


    t0 = time.time()
    to_delete.append("/v1/datasets/training_dataset_" + run_id)
    mldb2.put("/v1/datasets/training_dataset_" + run_id, {
        "type": "merged",
        "params": {
            "datasets": [
                {"id": "training_labels_" + run_id},
                {"id": embeddingDataset}
            ]
        }
    })
    t1 = time.time()
    times["4 - create merge labels + embedding"] = t1-t0

    modelDir = os.path.join(mldb.plugin.get_plugin_dir(), "models")
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)

    modelAbsolutePath = modelDir+"/dataset_creator_%s.cls.gz" % run_id

    mldb.log("Training with %d bags, %0.2f rnd feats" % (numBags, numRndFeats))

    t0 = time.time()
    to_delete.append("/v1/procedures/trainer_" + run_id)
    if not doDeploy:
        to_delete.append("/v1/functions/explorator_cls_" + run_id)
    rez = mldb2.put("/v1/procedures/trainer_" + run_id, {
        "type": "classifier.train",
        "params": {
            "trainingData": """
                SELECT {* EXCLUDING(weight, label)} as features,
                       weight AS weight,
                       label = 0 AS label
                FROM training_dataset_%s
                WHERE label IS NOT NULL
            """ % run_id,
            "modelFileUrl": "file://"+modelAbsolutePath,
            "algorithm": "my_bdt",
            "configuration": {
                "my_bdt": {
                    "type": "bagging",
                    "verbosity": 3,
                    "weak_learner": {
                        "type": "decision_tree",
                        "verbosity": 0,
                        "max_depth": 10,
                        "random_feature_propn": numRndFeats,
                    },
                    "num_bags": numBags
                }
            },
            "mode": "boolean",
            "functionName": cls_func_name
        }
    })
    t1 = time.time()
    times["5 - training"] = t1-t0


    t0 = time.time()
    query = """
        SELECT %s({features: {*}}) as *
        FROM %s
        ORDER BY score DESC""" % (cls_func_name, embeddingDataset)
    scores = mldb2.query(query)
    del scores[0]   # remove header
    num_images = len(scores)
    scores_dict = {v[0]: v[1] for v in scores}
    t1 = time.time()
    times["6 - scrore all examples"] = t1-t0


    if doDeploy:
        mldb2.put("/v1/procedures/transformer", {
            "type": "transform",
            "params": {
                "inputData": """
                    SELECT *, '%s' as datasetName, '%s' as imagePrefix
                    NAMED training_labels.rowName()
                    FROM training_labels_%s as training_labels
                    JOIN (
                        (
                         SELECT %s({features: {*}}) as *
                            FROM %s
                        ) as score
                    ) ON score.rowName() = training_labels.rowName()
                """ % (datasetName, prefix, run_id, cls_func_name, embeddingDataset),
                "outputDataset": "predictions_%s" % run_id
            }
        })


    t0 = time.time()
    already_added = groups[0].union(groups[1])

    grpA = [ (x, scores_dict[x])   for x in groups[0] ]
    grpB = [ (x, 1-scores_dict[x]) for x in groups[1] ]

    numImgToBreak = min(10, int(len(scores) / 2))

    exploitA = []
    exploitB = []
    for x in scores:
        if len(exploitA) >= numImgToBreak: break
        if x[0] in already_added: continue
        already_added.add(x[0])
        exploitA.append(x)

    for x in reversed(scores):
        if len(exploitB) >= numImgToBreak: break
        if x[0] in already_added: continue
        already_added.add(x[0])
        exploitB.append((x[0], 1-x[1]))


    t1 = time.time()
    times["7 - rest"] = t1-t0

    t0 = time.time()
    new_sample = []
    for elem in mldb2.query("select rowName() from sample(%s, {rows:%d})" % (embeddingDataset, min(100, num_images)))[1:]:
        if elem[0] in already_added: continue
        new_sample.append([elem[0],scores_dict[elem[0]]])
    t1 = time.time()
    times["8 - sample"] = t1-t0


    # house keeping
    for toDel in to_delete:
        mldb.log("    deleting " + toDel)
        mldb2.delete(toDel)
    if not doDeploy:
        mldb.log("    deleting " + modelAbsolutePath)
        os.remove(modelAbsolutePath)

    rtn_dict = {
        "a": {
            "prev": grpA,
            "exploit": [],
            "explore": exploitA
        },
        "b": {
            "prev": grpB,
            "exploit": [],
            "explore": exploitB,
        },
        "ignore": [],
        "sample": new_sample[:20],
        "deploy_id": run_id if doDeploy else ""
    }

    mldb.log(times)

    return (rtn_dict, 200)



def createDataset():
    import base64, re, os

    unique_id = str(binascii.hexlify(os.urandom(16)))
    payload = json.loads(mldb.plugin.rest_params.payload)

    collectionName = "dataset_" + payload['dataset'].replace(".", "").replace("/", "").replace("-", "_") + "_" + unique_id
    collectionFolder = os.path.join(mldb.plugin.get_plugin_dir(), "static", collectionName)
    if not os.path.exists(collectionFolder):
        os.mkdir(collectionFolder)

    # save images on disk
    for image in payload["images"]:
        imageName = image[0].lower().replace("/", "").replace("jpeg", "jpg")
        writer = open(os.path.join(collectionFolder, imageName), "w")

        imgstr = re.search(r'base64,(.*)', image[1]).group(1)
        writer.write(base64.decodestring(imgstr))
        writer.close()


    # embed folder
    payload = {
        "name": collectionName,
        "folder": collectionFolder,
        "limit": payload['limit']
    }
    mldb.log("calling embedding function")
    embedFolderWithPayload(payload)

    # create nearest neighbour function. this will allow us to quickly find similar images
    mldb2.put("/v1/functions/nearest_%s" % collectionName, {
        "type": "embedding.neighbors",
        "params": {
            "dataset": "embedded_images_%s" % collectionName
        }
    })

    return (payload, 200)


######
# The following is to embed images in a folder
def embedFolder():
    payload = json.loads(mldb.plugin.rest_params.payload)
    embedFolderWithPayload(payload)

def embedFolderWithPayload(payload):
    # create dataset with available images
    mldb.log("Creating dataset...")

    dataset_config = {
        'type'    : 'sparse.mutable',
        'id'      : payload["name"]
    }

    if "name" not in payload or "folder" not in payload:
        return ("missing keys!", 400)

    mldb2.delete("/v1/datasets/" + payload["name"])
    dataset = mldb.create_dataset(dataset_config)
    now = datetime.datetime.now()

    limit = -1
    if "limit" in payload:
        limit = payload["limit"]

    if "folder" not in payload:
        raise Exception("Folder must be specified!")

    # if we're loading images from disk
    for num_images, filename in enumerate(os.listdir(payload["folder"])):
        if limit>0 and num_images+1 > limit:
            break

        mldb.log(" .%d : %s" % (num_images, filename))
        dataset.record_row(filename.split(".")[0],
                            [["location", os.path.join(payload["folder"], filename), now]])
    dataset.commit()

    # now embed images
    mldb2.put("/v1/procedures/embedder", {
        "type": "transform",
        "params": {
            "inputData": """
                SELECT inception({url: location}) AS *
                FROM %s
            """ % payload["name"],
            "outputDataset": {
                "id": EMBEDDING_DATASET + "_" + payload["name"],
                "type": "embedding"
            }
        }
    })

    rtnVal = {
        "source": payload["folder"],
        "name": payload["name"],
        "num_images": num_images + 1
    }
    return (rtnVal, 200)


def persistEmbedding():
    payload = json.loads(mldb.plugin.rest_params.payload)

    outputFolder = os.path.join(mldb.plugin.get_plugin_dir(), "cache")
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    mldb2.post("/v1/procedures", {
        "type": "export.csv",
        "params": {
            "exportData": "select rowName() as rowName, * from %s_%s " % (EMBEDDING_DATASET, payload["name"]),
            "dataFileUrl": "file://"+os.path.join(outputFolder,
                                                  "dataset_creator_embedding_%s.csv.gz" % payload["name"]),
            "headers": True
        }
    })


    mldb2.post("/v1/procedures", {
        "type": "export.csv",
        "params": {
            "exportData": "select rowName() as rowName, * from %s" % payload["name"],
            "dataFileUrl": "file://"+os.path.join(outputFolder,
                                                  "dataset_creator_images_%s.csv.gz" % payload["name"]),
            "headers": True
        }
    })

    return ("Persisted!", 200)



####
# Handle an incoming request
####
msg = "Unknown route: " + mldb.plugin.rest_params.verb + " " + mldb.plugin.rest_params.remaining
rtnCode = 400

if mldb.plugin.rest_params.verb == "POST":
    if mldb.plugin.rest_params.remaining == "/similar":
        (msg, rtnCode) = getSimilar()
    elif mldb.plugin.rest_params.remaining == "/embedFolder":
        (msg, rtnCode) = embedFolder()
    elif mldb.plugin.rest_params.remaining == "/persistEmbedding":
        (msg, rtnCode) = persistEmbedding()
    elif mldb.plugin.rest_params.remaining == "/prediction":
        (msg, rtnCode) = getPrediction()
    elif mldb.plugin.rest_params.remaining == "/createDataset":
        (msg, rtnCode) = createDataset()

mldb.plugin.set_return(msg, rtnCode)


