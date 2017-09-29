#
# routes.py
# Francois Maillet, 2016-02-25
# Copyright (c) 2016 mldb.ai inc. All rights reserved.
#

import json, os, datetime, math
from operator import itemgetter
import functools, urllib, time
import binascii

if None:
    mldb = None  # mute pep8
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


    groups = [set(data["a"]), set(data["b"])]
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
        # TODO: unclear what happens on invalid img
        # TODO FMLH===========
#             "where" : "ResponseException: Response status code: 400. Response text: {\"details\":{\"entry\":\"embedder\",\"runError\":{\"details\":{\"
# calc\":[\"rowPath()\"],\"context\":{\"details\":null,\"error\":\"Unable to run model: image must have 1 or 3 channels, got [512,512,4]\\n\\t [[Node: op = EncodeJpeg[chrom
# a_downsampling=true, density_unit=\\\"in\\\", format=\\\"\\\", optimize_size=false, progressive=false, quality=95, x_density=300, xmp_metadata=\\\"\\\", y_density=300, _d
# evice=\\\"/job:localhost/replica:0/task:0/cpu:0\\\"](_recv_image_0)]]\"},\"from\":{\"columnCount\":1048577,\"rowCount\":157},\"limit\":-1,\"offset\":0,\"orderBy\":\"\",\"
# select\":\"CASE\\n                            WHEN content IS NOT NULL\\n                                THEN {content: content}\\n                            WHEN png_co
# ntent IS NOT NULL\\n                                THEN {content: tf_EncodeJpeg(png_content)}\\n                            ELSE\\n                                {conte
# nt: NULL}\\n                            END\",\"where\":\"true\"},\"error\":\"Execution error: Unable to run model: image must have 1 or 3 channels, got [512,512,4]\\n\\t
#  [[Node: op = EncodeJpeg[chroma_downsampling=true, density_unit=\\\"in\\\", format=\\\"\\\", optimize_size=false, progressive=false, quality=95, x_density=300, xmp_metada
# ta=\\\"\\\", y_density=300, _device=\\\"/job:localhost/replica:0/task:0/cpu:0\\\"](_recv_image_0)]]\",\"httpCode\":400}},\"error\":\"failed to create the initial run\",\"
# httpCode\":400}\n"
        # ============================================
        score_query_rez = mldb2.query("""
            SELECT score, prob_%s({score}) as *
            FROM (
                SELECT scorer_%s({features: {*}})[score] as score
                FROM (
                    SELECT inceptionJpeg({{content}}) as * FROM (
                        SELECT CASE
                            WHEN content IS NOT NULL
                                THEN {content: content}
                            WHEN png_content IS NOT NULL
                                THEN {content: tf_EncodeJpeg(png_content)}
                            ELSE
                                {content: NULL}
                            END
                        FROM (
                            SELECT CASE
                                WHEN regex_search(mime, 'JPEG')
                                    THEN {content: content}
                                WHEN regex_search(mime, 'PNG')
                                    THEN {png_content: tf_DecodePng(content)}
                                ELSE
                                    {content: NULL}
                                END AS *
                            FROM (
                                SELECT content, mime_type(content) AS mime FROM (
                                    SELECT fetcher('%s') AS *
                                ) WHERE error IS NULL
                            )
                        )
                    )
                )
            )
        """ % (input_data["deploy_id"], input_data["deploy_id"], input_data["image_url"]))
    except Exception as e:
        if unableToGetMimeType:
            return ("Error when trying to score image. Could not determine mime type. Probably not a JPEG image", 400)

        return ("Error scoring image: %s. URL: %s" % (str(e), input_data["image_url"]), 400)

    mldb.log(score_query_rez)
    score = score_query_rez[1][score_query_rez[0].index("score")]
    prob = score_query_rez[1][score_query_rez[0].index("prob")]


    return_val = {}

    mldb.log(input_data)

    if input_data["same_deploy"] == "false":
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

        return_val["example_image"] = example_images

    return_val["score"] = score
    return_val["prob"] =  prob


    return (return_val, 200)





def getSimilar():

    data, groups, doDeploy, datasetName, prefix, numBags, numRndFeats = preProcessData()

    embeddingDataset = EMBEDDING_DATASET + "_" + datasetName

    run_id = str(binascii.hexlify(os.urandom(16)))
    cls_func_name = "scorer_" + run_id
    prob_func_name = "prob_" + run_id

    # keep track of ressources to delete
    to_delete = []

    dataset_config = {
        'type'    : 'sparse.mutable',
        'id'      : "training_labels_pos_"+run_id #"training_labels_pos_" + run_id if not doDeploy else "training_labels_"+run_id
    }

    to_delete.append("/v1/datasets/" + dataset_config["id"])
    dataset = mldb.create_dataset(dataset_config)
    now = datetime.datetime.now()


    to_add = []
    already_added = set()

    num_posex = len(groups[0])
    if   num_posex > 20: posex_weight = 0.001
    elif num_posex > 10: posex_weight = 0.01
    else:                posex_weight = 0.05

    times = {}
    t0 = time.time()
    for lbl, imgs in enumerate(groups):
        for img in imgs:
            if img in already_added: continue
            dataset.record_row(img, [["label", lbl, now], ["weight", 1, now]])
            already_added.add(img)

            # if it's the positive group, look at nearest neighbhours to add extra
            # labels in positive class in case we don't have enough
            if lbl == 0: # and not doDeploy:
                # will return list of ["slid","596e1ca6687cd301",1.9799363613128662]
                neighbours = mldb2.query("select nearest_%s({coords: '%s'})[neighbors] as *" % (datasetName, img))

                for nName in neighbours[1][1:]:
                    to_add.append((nName, [["label", lbl, now], ["weight", posex_weight, now]]))


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
    if True: #not doDeploy:
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

    modelAbsolutePath = modelDir+"/deepteach_cls_%s.cls.gz" % run_id

    mldb.log("Training with %d bags, %0.2f rnd feats" % (numBags, numRndFeats))

    t0 = time.time()
    to_delete.append("/v1/procedures/trainer_" + run_id)
    if not doDeploy:
        to_delete.append("/v1/functions/scorer_" + run_id)

    if False:
        rez = mldb2.put("/v1/procedures/trainer_" + run_id, {
            "type": "randomforest.binary.train",
            "params": {
                "trainingData": """
                    SELECT {* EXCLUDING(weight, label)} as features,
                           weight AS weight,
                           label = 0 AS label
                    FROM training_dataset_%s
                    WHERE label IS NOT NULL
                """ % run_id,
                "modelFileUrl": "file://"+modelAbsolutePath,
                #"featureVectorSamplings": <int>,
                #"featureVectorSamplingProp": <float>,
                #"featureSamplings": <int>,
                "featureSamplingProp": numRndFeats,
                "maxDepth": 10,
                "functionName": cls_func_name
            }
        })
    else:
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
    lbl_count = mldb2.query("""
            SELECT count(*)
            FROM training_dataset_%s
            WHERE label IS NOT NULL
            GROUP BY label""" % run_id)
    pos_cnt = lbl_count[1][1]
    neg_cnt = lbl_count[2][1]
    num_images = float(pos_cnt+neg_cnt)

    to_delete.append("/v1/datasets/prob_train_%s" % run_id)
    mldb2.post("/v1/procedures", {
        "type": "transform",
        "params": {
            "inputData": """
                SELECT %s({features: {*}})[score] as score,
                       label = 0 AS label,
                       CASE label
                        WHEN 1 THEN %0.5f
                        ELSE %0.5f
                       END as weight
                FROM training_dataset_%s
                WHERE label IS NOT NULL
            """ % (cls_func_name, pos_cnt/num_images, neg_cnt/num_images, run_id),
            "outputDataset": "prob_train_%s" % run_id
        }
    })

    probAbsolutePath = modelDir+"/deepteach_prob_%s.prob.gz" % run_id
    mldb2.post("/v1/procedures", {
        "type": "probabilizer.train",
        "params": {
            "trainingData": """
                select score, label, weight from prob_train_%s
            """ % run_id,
            "modelFileUrl": "file://"+probAbsolutePath,
            "link": "COMP_LOG_LOG",
            "functionName": prob_func_name,
        }
    })
    t1 = time.time()
    times["? - train prob"] = t1-t0


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


    mldb2.put("/v1/procedures/transformer", {
        "type": "transform",
        "params": {
            "inputData": """
                SELECT score, prob_%s({score}) as *
                FROM (
                    SELECT scorer_%s({features: {*}})[score] as score
                    FROM %s
                )
            """ % (run_id, run_id, embeddingDataset),
            "outputDataset": "predictions_prob_%s" % run_id
        }
    })

    scores = mldb2.query("SELECT prob FROM predictions_prob_%s ORDER BY prob DESC" % run_id)
    del scores[0]
    scores_dict = {v[0]: v[1] for v in scores}
    mldb.log(scores_dict)

    grpA = [ (x, scores_dict[x]) for x in groups[0] ]
    grpB = [ (x, scores_dict[x]) for x in groups[1] ]

    numImgToBreak = min(10, int(len(scores) / 2))

    splitInHalf = True
    try:
        half_idx = next(x[0] for x in enumerate(scores) if x[1][1] < 0.5)
    except:
        # the COMP_LOG_LOG prob link function gives us a smoother function
        # from the negatives to the pos, but when there aren't a log of negs
        # it might not go under 50%. this is to compensate
        splitInHalf = False
        half_idx = len(scores) / 2


    exploitA = []
    exploitB = []
    unsure = []

    mldb.log(" ================ adding in a")
    for x in scores:
        if x[1] < 0.5: break
        if len(exploitA) >= numImgToBreak: break
        if x[0] in already_added: continue
        already_added.add(x[0])
        exploitA.append(x)
        mldb.log(x)

    mldb.log(" ================ adding in unsure")

    for x in scores[:half_idx]:
        if x[1] < 0.5: break
        if len(unsure) >= numImgToBreak/2: break
        if x[0] in already_added: continue
        already_added.add(x[0])
        unsure.append(x)
        mldb.log(x)

    mldb.log(" ================ adding in b")
    for x in reversed(scores):
        if splitInHalf and x[1] > 0.5: break
        if len(exploitB) >= numImgToBreak: break
        if x[0] in already_added: continue
        already_added.add(x[0])
        exploitB.append(x)
        mldb.log(x)

    mldb.log(" ================ adding in unsure")

    for x in scores[half_idx:]:
        if splitInHalf and x[1] > 0.5: break
        if len(unsure) >= numImgToBreak: break
        if x[0] in already_added: continue
        already_added.add(x[0])
        unsure.append(x)
        mldb.log(x)


    t1 = time.time()
    times["7 - rest"] = t1-t0

    t0 = time.time()
    for elem in mldb2.query("select rowName() from sample(%s, {rows:%d})" % (embeddingDataset, min(100, num_images)))[1:]:
        if elem[0] in already_added: continue
        if len(unsure) >= 20: break
        unsure.append([elem[0],scores_dict[elem[0]]])
    t1 = time.time()
    times["8 - sample"] = t1-t0


    # house keeping
    for toDel in to_delete:
        mldb.log("    deleting " + toDel)
        mldb2.delete(toDel)
    if not doDeploy:
        mldb.log("    deleting " + modelAbsolutePath)
        os.remove(modelAbsolutePath)
        mldb.log("    deleting " + probAbsolutePath)
        os.remove(probAbsolutePath)

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
        "sample": unsure,
        "deploy_id": run_id if doDeploy else ""
    }

    mldb.log(times)

    return (rtn_dict, 200)

def recreate_dataset():
    mldb.log("RECREATING DATASET")
    payload = json.loads(mldb.plugin.rest_params.payload)
    mldb.log(payload)

    unique_id = str(binascii.hexlify(os.urandom(16)))
    collection_name = "dataset_" + payload['dataset'].replace(".", "").replace("/", "").replace("-", "_") + "_" + unique_id
    collection_folder = os.path.join(mldb.plugin.get_plugin_dir(), "static", collection_name)
    if not os.path.exists(collection_folder):
        os.mkdir(collection_folder)

    # TODO DELETE
    dataset = mldb2.create_dataset({
        'id': 'deepteach_loader_tmp',
        'type': 'sparse.mutable'
    })
    for idx, image in enumerate(payload["images"]):
        if not image:
            continue
        if image[-1] == '\r':
            image = image[:-1]
        dataset.record_row(idx, [['image', image, 0]])
    dataset.commit()

    res = mldb2.query("SELECT image, fetcher(image) FROM deepteach_loader_tmp")
    first = True
    image_url_idx = 3
    image_data_idx = 1
    for idx, row in enumerate(res):
        if first:
            mldb.log(row)
            assert row[0] == '_rowName', row[0]
            assert row[1] == 'fetcher(image).content', row[1]
            assert row[2] == 'fetcher(image).error', row[2]
            assert row[image_url_idx] == 'image', row[image_url_idx]
            first = False
            continue

        lower = row[image_url_idx].lower()
        if lower.endswith(".png"):
            image_name = '{}.png'.format(idx)
        elif lower.endswith(".jpg") or lower.endswith('.jpeg'):
            image_name = '{}.jpg'.format(idx)
        else:
            raise Exception("Wrong extension: {}".format(lower))

        mldb2.log("URL: {}".format(row[image_url_idx]))
        mldb2.log("NAME: {}".format(image_name))
        blob = row[image_data_idx]['blob']

        with open(collection_folder + '/' + image_name, 'wb') as f:
            for idx in range(len(blob)):
                if type(blob[idx]) is int:
                    f.write(chr(blob[idx]))
                elif type(blob[idx]) is unicode:
                    f.write(blob[idx])
                else:
                    assert False, "should not be here"

    payload = {
        "name": collection_name,
        "folder": collection_folder,
        "limit": payload['limit']
    }

    mldb.log("calling embedding function")
    embedFolderWithPayload(payload)

    mldb2.put("/v1/functions/nearest_%s" % collection_name, {
        "type": "embedding.neighbors",
        "params": {
            "dataset": "embedded_images_%s" % collection_name
    }
})

    return (payload, 200)

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
        if limit > 0 and num_images + 1 > limit:
            break

        mldb.log(" .%d : %s" % (num_images, filename))
        dataset.record_row(filename.split(".")[0],
                            [["location", os.path.join(payload["folder"], filename), now]])
    dataset.commit()

    # now embed images
    # TODO don't drop the errors into the void
    mldb2.put("/v1/procedures/embedder", {
        "type": "transform",
        "params": {
            "inputData": """
                SELECT * FROM (
                    SELECT try(inceptionJpeg({{content}}), NULL) FROM (
                        SELECT CASE
                            WHEN content IS NOT NULL
                                THEN {{content: content}}
                            WHEN png_content IS NOT NULL
                                THEN {{content: tf_EncodeJpeg(png_content)}}
                            ELSE
                                {{content: NULL}}
                            END
                        FROM (
                            SELECT CASE
                                WHEN regex_search(mime, 'JPEG')
                                    THEN {{content: content}}
                                WHEN regex_search(mime, 'PNG')
                                    THEN {{png_content: tf_DecodePng(content)}}
                                ELSE
                                    {{content: NULL}}
                                END AS *
                            FROM (
                                SELECT content, mime_type(content) AS mime FROM (
                                    SELECT fetcher(location) AS *
                                    FROM {}
                                ) WHERE error IS NULL
                            )
                        )
                    ) WHERE content IS NOT NULL
                ) WHERE incept.pool_3 IS NOT NULL
            """.format(payload["name"]),
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
    elif mldb.plugin.rest_params.remaining == "/recreateDataset":
        (msg, rtnCode) = recreate_dataset()

mldb.plugin.set_return(msg, rtnCode)


