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


@dataclass
class ServiceRecord:
    timestamp: float
    service_id: int
    provider: int
    requester: int
    duration: float


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
    record_id: int,
) -> CapacityRecord:

    dynamic_price = server_vehicle.price_init * server_vehicle.quality

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

    return record


def add_to_pending_capacity_records(record: CapacityRecord):

    pending_capacity_records.append(record)

    print(f"[Record {record.record_id}] " f"Added to pending capacity records.")


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

    print(
        f"[MEC Side] Service record created: "
        f"Requester {service_record.requester} -> Provider {service_record.provider}"
    )

    return service_record


def add_to_pending_service_records(service_record: ServiceRecord):

    pending_service_records.append(service_record)

    print(
        f"[Service Record {service_record.service_id}] "
        f"Added to pending service records."
    )


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
    new_record_id: int,
) -> CapacityRecord | None:

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

    print(
        f"[Capacity Update] New capacity record created for "
        f"Provider {provider_id} with remaining period {updated_period:.6f}."
    )

    return updated_record
