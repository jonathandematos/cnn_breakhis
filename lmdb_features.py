#!/usr/bin/python
#
import lmdb
import sys
#
database = sys.argv[1]
features = sys.argv[2]
#
db_env = lmdb.open(database, map_size=int(1e9), readonly=False)
file_feat = open(features, "r")
#
with db_env.begin(write=True) as db_handler:
    #
    for i in file_feat:
        indexed = i[:-1].split(";")
        key = indexed[1]
        value = i[i.find(";") + len(key) + 2:-2]
        print(value)
        db_handler.put(key, value)
#
db_env.close()
file_feat.close()

