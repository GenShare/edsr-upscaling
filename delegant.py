#!.venv/bin/python
import json
import json.decoder
import logging
import os
import subprocess
import sys
import time

from os import environ
from threading import Thread

import boto3
import botocore

# TODO: break out these functions from upscaleFromPb
from run import load_pb, run_from_pb

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# TODO: update to _path_from_content based on hash
def _path_from_uid(uid: str) -> str:
    return f'/artifacts/{uid}.png'

def process_queue(thread_id, sqs):
    while True:
        response = sqs.receive_message(QueueUrl=environ['INBOUND_REQUESTS_QUEUE_URL'],
                WaitTimeSeconds=20)

        if 'Messages' in response:
            for message in response['Messages']:
                uid, body = message['MessageId'], message['Body']
                sqs.delete_message(
                        QueueUrl=environ['INBOUND_REQUESTS_QUEUE_URL'],
                        ReceiptHandle=message['ReceiptHandle']
                )

                # TODO: accept artifact_id instead of prompt
                # artifact should be at "/artifacts/{id}"

                try:
                    body = json.loads(body)
                    prompt = body['prompt']
                except json.decoder.JSONDecodeError:
                    logging.info(f'Unable to parse {uid}: "{body}"')
                    continue
                except KeyError:
                    logging.info(f'{uid}: no prompt')
                    continue

                logging.info(f'RECV {uid}: "{prompt}"')

                # TODO: parse addtl params

                # TODO: pass required args
                run_from_pb(prompt, model)

def main():
    sqs = boto3.client('sqs', region_name=environ['AWS_REGION'])

    # TODO: update this invocation and pass results to Thread for runs
    pb = load_pb()

    threads = [
                Thread(target=process_queue, args=(i+1, sqs)
                for i in range(int(environ['N_THREADS']))
              ]

    for thread in threads:
        thread.start()

if __name__ == '__main__':
    main()