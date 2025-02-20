from api.sender import SenderTCP

import uuid

uudis = str(uuid.uuid4())

sender = SenderTCP("172.16.103.17", 8080)

sender.send_request(machine_id=0, uuid=0)