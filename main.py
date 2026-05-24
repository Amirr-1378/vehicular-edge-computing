from mec_side import (
    ServerVehicle,
    UserVehicle,
    add_block_to_capacity_chain,
    add_to_pending_capacity_records,
    broadcast_capacity_info,
    capacity_chain,
    consensus_process,
    create_capacity_record,
    pending_capacity_records,
    verify_certificate,
)
from user_side import receive_capacity_info, select_available_candidate

# =========================================================
# BLOCK 21: Main Flow Initialization
# =========================================================


def main():

    print("Simulation started.")

    # =========================================================
    # BLOCK 22: Simulation Entities and Initial Parameters
    # =========================================================

    server_vehicles = [
        ServerVehicle(
            vehicle_id=1,
            resource=2e9,
            trajectory=120.0,
            price_init=0.6,
            period=60.0,
            quality=0.9,
        ),
        ServerVehicle(
            vehicle_id=2,
            resource=1.5e9,
            trajectory=130.0,
            price_init=0.7,
            period=50.0,
            quality=0.85,
        ),
    ]

    user_vehicle = UserVehicle(
        vehicle_id=101,
        trajectory=118.0,
    )

    print("[Main] Server vehicles initialized.")
    print("[Main] User vehicle initialized.")

    # =========================================================
    # BLOCK 23: Capacity Record Creation and Capacity Chain Update
    # =========================================================

    record_id = 1

    for server_vehicle in server_vehicles:

        if verify_certificate(server_vehicle):

            capacity_record = create_capacity_record(
                server_vehicle=server_vehicle,
                record_id=record_id,
            )

            add_to_pending_capacity_records(capacity_record)

            record_id += 1

    if consensus_process(pending_capacity_records):
        add_block_to_capacity_chain(pending_capacity_records)

    print("\n[Main] Capacity Chain:")
    print(capacity_chain)

    # =========================================================
    # BLOCK 24: Broadcast Capacity Information to User Vehicle
    # =========================================================

    broadcasted_capacity_info = broadcast_capacity_info()

    received_capacity_info = receive_capacity_info(broadcasted_capacity_info)

    print("\n[Main] Received Capacity Information:")
    print(received_capacity_info)

    # =========================================================
    # BLOCK 25: Select Available Candidate Server Vehicle
    # =========================================================

    selected_candidate = select_available_candidate(
        user_vehicle=user_vehicle,
        capacity_records=received_capacity_info,
        service_records=[],
    )

    print("\n[Main] Selected Candidate:")
    print(selected_candidate)


if __name__ == "__main__":
    main()
