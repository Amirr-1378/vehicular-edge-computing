# =========================================================
# BLOCK 1: Vehicular MEC System Data Models
# =========================================================

from dataclasses import dataclass


@dataclass
class ServerVehicle:
    vehicle_id: int
    resource: float
    trajectory: float
    price_init: float
    period: float
    quality: float
    certificate_exists: bool = True
    certificate_is_new: bool = True
    certificate_is_valid: bool = True


@dataclass
class UserVehicle:
    vehicle_id: int
    trajectory: float


@dataclass
class Task:
    user_id: int
    input_size: float
    complexity: float
    deadline: float


# =========================================================
# BLOCK 53: Digital Signature Infrastructure
# =========================================================
@dataclass
class Certificate:

    owner_id: int
    public_key: str
    issuer: str


@dataclass
class CapacityRecord:
    timestamp: float
    record_id: int
    provider: int
    resource: float
    trajectory: float
    price: float
    period: float
    quality: float
    # =========================================================
    # BLOCK 57: Add Certificate and Signature to CapacityRecord
    # =========================================================
    certificate: Certificate | None = None
    signature: str = ""


@dataclass
class ServiceRecord:
    timestamp: float
    service_id: int
    provider: int
    requester: int
    duration: float
    # =========================================================
    # BLOCK 60: Add Certificate and Signature to ServiceRecord
    # =========================================================
    certificate: Certificate | None = None
    signature: str = ""


# =========================================================
# BLOCK 42: MEC Node Model for Proof of Service
# =========================================================
@dataclass
class MECNode:
    node_id: int
    redundant_resource: float
    is_leader: bool = False


# =========================================================
# BLOCK 2: Certificate Verification and Capacity Record Creation
# =========================================================

import time

pending_capacity_records = []


def verify_certificate(server_vehicle: ServerVehicle) -> bool:

    if not server_vehicle.certificate_exists:
        print(f"[Vehicle {server_vehicle.vehicle_id}] Certificate does not exist.")
        return False

    if not server_vehicle.certificate_is_new:
        print(f"[Vehicle {server_vehicle.vehicle_id}] Certificate is not new.")
        return False

    if not server_vehicle.certificate_is_valid:
        print(f"[Vehicle {server_vehicle.vehicle_id}] Certificate is invalid.")
        return False

    print(f"[Vehicle {server_vehicle.vehicle_id}] Certificate verified successfully.")
    return True


def create_capacity_record(
    server_vehicle: ServerVehicle,
) -> CapacityRecord:

    dynamic_price = server_vehicle.price_init * server_vehicle.quality

    record_id = generate_capacity_record_id()

    record = CapacityRecord(
        timestamp=time.time(),
        record_id=record_id,
        provider=server_vehicle.vehicle_id,
        resource=server_vehicle.resource,
        trajectory=server_vehicle.trajectory,
        price=dynamic_price,
        period=server_vehicle.period,
        quality=server_vehicle.quality,
    )

    certificate = create_certificate(
        owner_id=server_vehicle.vehicle_id,
    )

    private_key = generate_private_key(
        server_vehicle.vehicle_id,
    )

    message = build_capacity_record_message(
        record,
    )

    signature = sign_message(
        message=message,
        private_key=private_key,
    )

    record.certificate = certificate
    record.signature = signature

    return record


def add_to_pending_capacity_records(record: CapacityRecord):

    pending_capacity_records.append(record)

    print(f"[Record {record.record_id}] " f"Added to pending capacity records.")


# =========================================================
# BLOCK 43: Leader Selection by Proof of Service
# =========================================================


def select_leader_by_pos(mec_nodes: list[MECNode]) -> MECNode:

    leader = max(mec_nodes, key=lambda node: node.redundant_resource)

    leader.is_leader = True

    print(
        f"[PoS] Leader selected: MEC {leader.node_id} "
        f"(resource={leader.redundant_resource})"
    )

    return leader


# =========================================================
# BLOCK 46: PBFT Pre-Prepare Phase
# =========================================================


def pre_prepare_phase(
    leader_node: MECNode,
    mec_nodes: list[MECNode],
    pending_records: list,
    next_block_id: int,
) -> dict | None:

    if len(pending_records) == 0:
        print("[Pre-Prepare] No pending records available.")
        return None

    proposed_block = {
        "block_id": next_block_id,
        "leader_id": leader_node.node_id,
        "records": pending_records.copy(),
    }

    print(
        f"[Pre-Prepare] Leader MEC {leader_node.node_id} "
        f"generated proposed Block {next_block_id}."
    )

    for node in mec_nodes:

        if node.node_id != leader_node.node_id:
            print(
                f"[Pre-Prepare] Proposed Block {next_block_id} "
                f"sent from Leader MEC {leader_node.node_id} "
                f"to MEC {node.node_id}."
            )

    return proposed_block


# =========================================================
# BLOCK 51: Real Proposed Block Validation Before Prepare
# =========================================================


def validate_proposed_block(
    proposed_block: dict | None,
    selected_leader: MECNode,
) -> bool:

    if proposed_block is None:
        print("[Validation] Proposed block is missing.")
        return False

    if "block_id" not in proposed_block:
        print("[Validation] Block ID is missing.")
        return False

    if "leader_id" not in proposed_block:
        print("[Validation] Leader ID is missing.")
        return False

    if "records" not in proposed_block:
        print("[Validation] Records are missing.")
        return False

    block_id = proposed_block["block_id"]
    leader_id = proposed_block["leader_id"]
    records = proposed_block["records"]

    if leader_id != selected_leader.node_id:
        print(
            f"[Validation] Invalid leader. "
            f"Expected MEC {selected_leader.node_id}, got MEC {leader_id}."
        )
        return False

    if block_id <= 0:
        print("[Validation] Invalid block ID.")
        return False

    if len(records) == 0:
        print("[Validation] Proposed block has no records.")
        return False

    for record in records:

        if record.record_id <= 0:
            print(f"[Validation] Invalid record ID in Record {record.record_id}.")
            return False

        if record.provider <= 0:
            print(f"[Validation] Invalid provider in Record {record.record_id}.")
            return False

        if record.resource <= 0:
            print(f"[Validation] Invalid resource in Record {record.record_id}.")
            return False

        if record.trajectory < 0:
            print(f"[Validation] Invalid trajectory in Record {record.record_id}.")
            return False

        if record.price < 0:
            print(f"[Validation] Invalid price in Record {record.record_id}.")
            return False

        if record.period < 0:
            print(f"[Validation] Invalid period in Record {record.record_id}.")
            return False

        if not (0 <= record.quality <= 1):
            print(f"[Validation] Invalid quality in Record {record.record_id}.")
            return False
        if record.certificate is None:
            print(f"[Validation] Missing certificate in Record {record.record_id}.")
            return False

        message = build_capacity_record_message(record)

        is_signature_valid = verify_signature(
            message=message,
            signature=record.signature,
            public_key=record.certificate.public_key,
        )

        if not is_signature_valid:
            print(f"[Validation] Invalid signature in Record {record.record_id}.")
            return False

    print(
        f"[Validation] Proposed Block {block_id} "
        f"validated successfully with {len(records)} signed record(s)."
    )

    return True


# =========================================================
# BLOCK 52: PBFT Consensus Threshold
# =========================================================


def calculate_pbft_threshold(
    total_nodes: int,
) -> int:

    faulty_nodes = (total_nodes - 1) // 3

    threshold = (2 * faulty_nodes) + 1

    print(
        f"[PBFT Threshold] Total nodes: {total_nodes}, "
        f"Faulty nodes tolerated: {faulty_nodes}, "
        f"Required votes: {threshold}"
    )

    return threshold


# =========================================================
# BLOCK 54: Mock Key Generation
# =========================================================


def generate_public_key(node_id: int) -> str:

    return f"PUBLIC_KEY_{node_id}"


def generate_private_key(node_id: int) -> str:

    return f"PRIVATE_KEY_{node_id}"


# =========================================================
# BLOCK 58: Certificate Generation
# =========================================================


def create_certificate(
    owner_id: int,
) -> Certificate:

    return Certificate(
        owner_id=owner_id,
        public_key=generate_public_key(owner_id),
        issuer="Trusted_MEC_CA",
    )


# =========================================================
# BLOCK 55: Mock Digital Signature
# =========================================================


def sign_message(
    message: str,
    private_key: str,
) -> str:

    return f"SIGN({message})"


def verify_signature(
    message: str,
    signature: str,
    public_key: str,
) -> bool:

    expected_signature = f"SIGN({message})"

    return signature == expected_signature


# =========================================================
# BLOCK 56: Capacity Record Message Builder
# =========================================================


def build_capacity_record_message(
    record: CapacityRecord,
) -> str:

    message = (
        f"{record.timestamp}|"
        f"{record.record_id}|"
        f"{record.provider}|"
        f"{record.resource}|"
        f"{record.trajectory}|"
        f"{record.price}|"
        f"{record.period}|"
        f"{record.quality}"
    )

    return message


# =========================================================
# BLOCK 61: Service Record Message Builder
# =========================================================


def build_service_record_message(
    record: ServiceRecord,
) -> str:

    message = (
        f"{record.timestamp}|"
        f"{record.service_id}|"
        f"{record.provider}|"
        f"{record.requester}|"
        f"{record.duration}"
    )

    return message


# =========================================================
# BLOCK 63: Service Record Validation
# =========================================================


def validate_service_record(
    record: ServiceRecord,
) -> bool:

    if record.service_id <= 0:
        print(f"[Service Validation] Invalid service ID: {record.service_id}.")
        return False

    if record.provider <= 0:
        print(f"[Service Validation] Invalid provider in Service {record.service_id}.")
        return False

    if record.requester <= 0:
        print(f"[Service Validation] Invalid requester in Service {record.service_id}.")
        return False

    if record.duration < 0:
        print(f"[Service Validation] Invalid duration in Service {record.service_id}.")
        return False

    if record.certificate is None:
        print(
            f"[Service Validation] Missing certificate in Service {record.service_id}."
        )
        return False

    message = build_service_record_message(record)

    is_signature_valid = verify_signature(
        message=message,
        signature=record.signature,
        public_key=record.certificate.public_key,
    )

    if not is_signature_valid:
        print(f"[Service Validation] Invalid signature in Service {record.service_id}.")
        return False

    print(
        f"[Service Validation] Service Record {record.service_id} "
        f"validated successfully."
    )

    return True


# =========================================================
# BLOCK 47: PBFT Prepare Phase
# =========================================================


def prepare_phase(
    mec_nodes: list[MECNode],
    proposed_block: dict | None,
    selected_leader: MECNode,
) -> list[dict]:

    prepare_messages = []

    if proposed_block is None:
        print("[Prepare] No proposed block received.")
        return prepare_messages

    block_id = proposed_block["block_id"]
    leader_id = proposed_block["leader_id"]
    records = proposed_block["records"]

    print(
        f"[Prepare] MEC nodes received proposed Block {block_id} "
        f"from Leader MEC {leader_id}."
    )

    for node in mec_nodes:

        # is_valid = validate_proposed_block(
        #     proposed_block=proposed_block,
        #     selected_leader=selected_leader,
        # )

        first_record = records[0]

        if isinstance(first_record, CapacityRecord):
            is_valid = validate_proposed_block(
                proposed_block=proposed_block,
                selected_leader=selected_leader,
            )

        elif isinstance(first_record, ServiceRecord):
            is_valid = all(validate_service_record(record) for record in records)

        else:
            print("[Prepare] Unknown record type.")
            is_valid = False

        if is_valid:

            prepare_message = {
                "node_id": node.node_id,
                "block_id": block_id,
                "vote": "PREPARE",
            }

            prepare_messages.append(prepare_message)

            print(
                f"[Prepare] MEC {node.node_id} verified Block {block_id} "
                f"and broadcasted PREPARE message."
            )

        else:

            print(f"[Prepare] MEC {node.node_id} rejected Block {block_id}.")

    print(f"[Prepare] Total PREPARE messages: {len(prepare_messages)}")

    return prepare_messages


# =========================================================
# BLOCK 48: PBFT Commit Phase
# =========================================================


def commit_phase(
    mec_nodes: list[MECNode],
    proposed_block: dict | None,
    prepare_messages: list[dict],
) -> list[dict]:

    commit_messages = []

    if proposed_block is None:
        print("[Commit] No proposed block received.")
        return commit_messages

    block_id = proposed_block["block_id"]

    required_votes = calculate_pbft_threshold(
        total_nodes=len(mec_nodes),
    )

    if len(prepare_messages) < required_votes:
        print(
            f"[Commit] Not enough PREPARE messages for Block {block_id}. "
            f"Required: {required_votes}, Received: {len(prepare_messages)}"
        )
        return commit_messages

    print(
        f"[Commit] Enough PREPARE messages received for Block {block_id}. "
        f"Required: {required_votes}, Received: {len(prepare_messages)}"
    )

    for node in mec_nodes:

        commit_message = {
            "node_id": node.node_id,
            "block_id": block_id,
            "vote": "COMMIT",
        }

        commit_messages.append(commit_message)

        print(
            f"[Commit] MEC {node.node_id} broadcasted COMMIT "
            f"message for Block {block_id}."
        )

    print(f"[Commit] Total COMMIT messages: {len(commit_messages)}")

    return commit_messages


# =========================================================
# BLOCK 49: PBFT Reply Phase
# =========================================================


def reply_phase(
    mec_nodes: list[MECNode],
    proposed_block: dict | None,
    commit_messages: list[dict],
) -> bool:

    if proposed_block is None:
        print("[Reply] No proposed block received.")
        return False

    block_id = proposed_block["block_id"]
    leader_id = proposed_block["leader_id"]

    required_votes = calculate_pbft_threshold(
        total_nodes=len(mec_nodes),
    )

    if len(commit_messages) < required_votes:
        print(
            f"[Reply] Not enough COMMIT messages for Block {block_id}. "
            f"Required: {required_votes}, Received: {len(commit_messages)}"
        )
        return False

    print(
        f"[Reply] Enough COMMIT messages received for Block {block_id}. "
        f"Required: {required_votes}, Received: {len(commit_messages)}"
    )

    for node in mec_nodes:

        print(
            f"[Reply] MEC {node.node_id} sent consensus result "
            f"to Leader MEC {leader_id} for Block {block_id}."
        )

    print(f"[Reply] Consensus result accepted by Leader MEC {leader_id}.")

    return True


# =========================================================
# BLOCK 50: PBFT Store Phase
# =========================================================


def store_phase(
    proposed_block: dict | None,
    consensus_result: bool,
    blockchain: list,
    pending_records: list,
) -> bool:

    if proposed_block is None:
        print("[Store] No proposed block to store.")
        return False

    if not consensus_result:
        print("[Store] Consensus failed. Block will not be stored.")
        return False

    block = {
        "block_id": proposed_block["block_id"],
        "leader_id": proposed_block["leader_id"],
        "records": proposed_block["records"],
    }

    blockchain.append(block)

    pending_records.clear()

    print(
        f"[Store] Block {block['block_id']} stored successfully "
        f"by Leader MEC {block['leader_id']}."
    )

    print("[Store] Pending records cleared after storing block.")

    return True


# =========================================================
# BLOCK 64: PBFT Consensus Wrapper
# =========================================================


def pbft_consensus_process(
    mec_nodes: list[MECNode],
    pending_records: list,
    blockchain: list,
) -> bool:

    if len(pending_records) == 0:
        print("[PBFT] No pending records to process.")
        return False

    leader_node = select_leader_by_pos(
        mec_nodes=mec_nodes,
    )

    proposed_block = pre_prepare_phase(
        leader_node=leader_node,
        mec_nodes=mec_nodes,
        pending_records=pending_records,
        next_block_id=len(blockchain) + 1,
    )

    prepare_messages = prepare_phase(
        mec_nodes=mec_nodes,
        proposed_block=proposed_block,
        selected_leader=leader_node,
    )

    commit_messages = commit_phase(
        mec_nodes=mec_nodes,
        proposed_block=proposed_block,
        prepare_messages=prepare_messages,
    )

    consensus_result = reply_phase(
        mec_nodes=mec_nodes,
        proposed_block=proposed_block,
        commit_messages=commit_messages,
    )

    store_result = store_phase(
        proposed_block=proposed_block,
        consensus_result=consensus_result,
        blockchain=blockchain,
        pending_records=pending_records,
    )

    print("\n[PBFT] Final PBFT Consensus Result:")
    print(store_result)

    return store_result


# =========================================================
# BLOCK 3: Consensus Process and Capacity Chain Update
# =========================================================

capacity_chain = []


def consensus_process(pending_records: list[CapacityRecord]) -> bool:

    if len(pending_records) == 0:
        print("[Consensus] No pending records to process.")
        return False

    print("[Consensus] Consensus process started.")
    print(f"[Consensus] Number of pending records: {len(pending_records)}")

    consensus_success = True

    if consensus_success:
        print("[Consensus] Consensus reached successfully.")
        return True

    print("[Consensus] Consensus failed.")
    return False


def add_block_to_capacity_chain(pending_records: list[CapacityRecord]):

    block_id = len(capacity_chain) + 1

    block = {
        "block_id": block_id,
        "records": pending_records.copy(),
    }

    capacity_chain.append(block)

    pending_records.clear()

    print(f"[Capacity Chain] Block {block_id} added successfully.")
    print("[Capacity Chain] Pending capacity records cleared.")


# =========================================================
# BLOCK 36: Automatic Capacity Record ID Generation
# =========================================================


def generate_capacity_record_id() -> int:

    confirmed_records_count = 0

    for block in capacity_chain:
        confirmed_records_count += len(block["records"])

    pending_records_count = len(pending_capacity_records)

    record_id = 1000 + confirmed_records_count + pending_records_count + 1

    return record_id


# =========================================================
# BLOCK 4: Broadcast Capacity Information to User Vehicles
# =========================================================


def broadcast_capacity_info():

    if len(capacity_chain) == 0:
        print("[Broadcast] Capacity chain is empty.")
        return []

    latest_block = capacity_chain[-1]

    print(f"[Broadcast] Broadcasting Block " f"{latest_block['block_id']} to vehicles.")

    records = latest_block["records"]

    for record in records:

        print(
            f"[Broadcast] Provider {record.provider} | "
            f"Resource: {record.resource} | "
            f"Price: {record.price}"
        )

    return records


# =========================================================
# BLOCK 16: Service Request Handling and Service Record Creation
# =========================================================

pending_service_records = []
service_chain = []

# =========================================================
# BLOCK 35: Automatic Service ID Generation
# =========================================================


def generate_service_id() -> int:

    confirmed_records_count = 0

    for block in service_chain:
        confirmed_records_count += len(block["records"])

    pending_records_count = len(pending_service_records)

    service_id = 2000 + confirmed_records_count + pending_records_count + 1

    return service_id


def create_service_record(
    service_request: dict,
) -> ServiceRecord:

    service_id = generate_service_id()

    service_record = ServiceRecord(
        timestamp=time.time(),
        service_id=service_id,
        provider=service_request["provider"],
        requester=service_request["requester"],
        duration=service_request["estimated_duration"],
    )
    certificate = create_certificate(
        owner_id=service_record.provider,
    )

    private_key = generate_private_key(
        service_record.provider,
    )

    message = build_service_record_message(
        service_record,
    )

    signature = sign_message(
        message=message,
        private_key=private_key,
    )

    service_record.certificate = certificate
    service_record.signature = signature

    print(
        f"[MEC Side] Service record created: "
        f"Requester {service_record.requester} -> Provider {service_record.provider}"
    )

    return service_record


def add_to_pending_service_records(service_record: ServiceRecord):
    if not validate_service_record(service_record):
        print(
            f"[Service Record {service_record.service_id}] "
            f"Rejected and not added to pending service records."
        )
        return
    pending_service_records.append(service_record)

    print(
        f"[Service Record {service_record.service_id}] "
        f"Added to pending service records."
    )


# =========================================================
# BLOCK 37: Get Confirmed Service Records from Service Chain
# =========================================================


def get_confirmed_service_records() -> list[ServiceRecord]:

    confirmed_service_records = []

    for block in service_chain:
        confirmed_service_records.extend(block["records"])

    return confirmed_service_records


# =========================================================
# BLOCK 17: Consensus Process and Service Chain Update
# =========================================================


def add_block_to_service_chain(pending_records: list[ServiceRecord]):

    block_id = len(service_chain) + 1

    block = {
        "block_id": block_id,
        "records": pending_records.copy(),
    }

    service_chain.append(block)

    pending_records.clear()

    print(f"[Service Chain] Block {block_id} added successfully.")
    print("[Service Chain] Pending service records cleared.")


# =========================================================
# BLOCK 18: Server Vehicle Task Execution
# =========================================================


def execute_task_on_server_vehicle(
    service_record: ServiceRecord,
) -> dict:

    execution_result = {
        "executor": f"Server Vehicle {service_record.provider}",
        "provider": service_record.provider,
        "requester": service_record.requester,
        "latency": service_record.duration,
        "status": "completed",
    }

    print(
        f"[Server Vehicle Execution] Provider {service_record.provider} "
        f"executed task for User {service_record.requester} "
        f"with latency {service_record.duration:.6f}."
    )

    return execution_result


# =========================================================
# BLOCK 19: Capacity Chain Update After Task Execution
# =========================================================


def update_capacity_after_execution(
    provider_id: int,
    used_duration: float,
):

    updated_records = []

    for block in capacity_chain:

        for record in block["records"]:

            if record.provider == provider_id:

                record.period = max(
                    0.0,
                    record.period - used_duration,
                )

                updated_records.append(record)

                print(
                    f"[Capacity Update] Provider {provider_id} "
                    f"remaining period updated to {record.period:.6f}."
                )

    if len(updated_records) == 0:
        print(
            f"[Capacity Update] No capacity record found "
            f"for Provider {provider_id}."
        )

    return updated_records


# =========================================================
# BLOCK 33: Create Updated Capacity Record After Execution
# =========================================================


def find_latest_capacity_record(
    provider_id: int,
) -> CapacityRecord | None:

    for block in reversed(capacity_chain):

        for record in reversed(block["records"]):

            if record.provider == provider_id:
                return record

    return None


def create_updated_capacity_record_after_execution(
    provider_id: int,
    used_duration: float,
) -> CapacityRecord | None:

    new_record_id = generate_capacity_record_id()

    latest_record = find_latest_capacity_record(
        provider_id=provider_id,
    )

    if latest_record is None:
        print(
            f"[Capacity Update] No previous capacity record found "
            f"for Provider {provider_id}."
        )
        return None

    updated_period = max(
        0.0,
        latest_record.period - used_duration,
    )

    updated_record = CapacityRecord(
        timestamp=time.time(),
        record_id=new_record_id,
        provider=latest_record.provider,
        resource=latest_record.resource,
        trajectory=latest_record.trajectory,
        price=latest_record.price,
        period=updated_period,
        quality=latest_record.quality,
    )

    certificate = create_certificate(
        owner_id=updated_record.provider,
    )

    private_key = generate_private_key(
        updated_record.provider,
    )

    message = build_capacity_record_message(
        updated_record,
    )

    signature = sign_message(
        message=message,
        private_key=private_key,
    )

    updated_record.certificate = certificate
    updated_record.signature = signature

    print(
        f"[Capacity Update] New capacity record created for "
        f"Provider {provider_id} with remaining period {updated_period:.6f}."
    )

    return updated_record
