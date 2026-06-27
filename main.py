import argparse
import time

from scripts import HelloBalls_Serial


def main():
    parser = argparse.ArgumentParser(description="HelloBalls RDK host entry point.")
    parser.add_argument("--serial-port", default="/dev/ttyS1", help="UART device path for the MCU.")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--print-imu", action="store_true", help="Print received IMU frames.")
    args = parser.parse_args()

    receiver = HelloBalls_Serial.SerialReceiver(port=args.serial_port, baudrate=args.baudrate)
    receiver.start()
    print(f"Serial receiver started on {args.serial_port}. Press Ctrl+C to stop.")

    try:
        while True:
            state = receiver.get_latest_state()
            if args.print_imu and state is not None:
                print(HelloBalls_Serial.format_state(state))
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping HelloBalls host.")
    finally:
        receiver.close()


if __name__ == "__main__":
    main()
