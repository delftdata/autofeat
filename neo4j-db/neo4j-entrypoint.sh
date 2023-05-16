#!/bin/bash

echo "====START====="
cypher-shell -d system "create database alldatamixed"
cypher-shell -d system "stop database alldatamixed"
neo4j-admin database load --from-path=/neo4j-db --overwrite-destination alldatamixed
chown -R neo4j:neo4j /data/databases/alldatamixed
chown -R neo4j:neo4j /data/transactions/alldatamixed
cypher-shell -d system "start database alldatamixed"
echo "======END======="
