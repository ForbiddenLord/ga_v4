# GA-STT — Crazyflie Firmware Integration

How to get the Crazyflie firmware and wire the GA-STT controller into it alongside PID, Mellinger, INDI, Brescianini, and Lee.

## 1. Get the firmware

```bash
git clone --recursive https://github.com/bitcraze/crazyflie-firmware.git
cd crazyflie-firmware
git checkout 2026.04
git submodule update --init --recursive
```

## 2. File mapping

This repo's `controller/` mirrors the firmware's controller directory. Merge
it in as follows:

| This repo | Firmware path | Status |
|---|---|---|
| `controller/controller.h` | `src/modules/interface/controller/controller.h` | **overwrite existing** |
| `controller/controller.c` | `src/modules/src/controller/controller.c` | **overwrite existing** |
| `controller/Kbuild` | `src/modules/src/controller/Kbuild` | **overwrite existing** |
| `controller/ga_stt/*` | `src/modules/src/controller/ga_stt/` | **new directory** |

## 3. Wiring the build (Kbuild)

The `src/modules/src/controller/ga_stt/Kbuild` file sets the C++ and Eigen compiler flags for that specific subdirectory, ensuring the rest of the (C) firmware remains unaffected

Because the Crazyflie firmware is cross-compiled for ARM, you cannot rely on system-wide Eigen installation. You must provide a local copy of the Eigen headers for the cross-compiler to find.

**Action needed:** Open `src/modules/src/controller/ga_stt/Kbuild` and replace `<path-to-eigen>` with the path to Eigen.

## 4. Build
```bash
cd <path-to-crazyflie-firmware>
make clean && make cf2_defconfig && make -j$(nproc)
```
## 5. flash
### Method 1. Radio OTA
Put Crazyflie in bootloader mode and run
```bash
make cload                           # flashes over radio
```

### Method 2. USB DFU
Power on normally, disconnect battery, plug usb and run
```bash
make flash_dfu                       # flashes over USB
``` 

## 6. Selecting GA-STT at runtime

GA-STT is index `6`. In the Crazyflie Python
client (cfclient):

1. Connect, open the **Parameters** tab.
2. Set `stabilizer.controller` to `6`.
3. Check the **Console** tab for:
   ```
   CONTROLLER: Using GA-STT (6) controller
   ```

## 7. Tunable parameters

Exposed under the `ga_stt` and `ga_stt_ctl` parameter groups:

| Group | Parameter | Meaning |
|---|---|---|
| `ga_stt` | `kappa1`, `kr`, `komega` | Controller gains |
| `ga_stt` | `eta_x/y/z` | STT tube attractor target |
| `ga_stt` | `rho_max` | Safety-tube radius |
| `ga_stt_ctl` | `decimation` | Run the full `compute_control()` every Nth tick (cached in between) |
| `ga_stt_ctl` | `relatch` | Re-latch the hover reference (`σ`, `R_d`) to the current state on the next tick |

## 8. Verifying the build

To build without installing ARM cross-compilers on your host machine, use the official Bitcraze Docker container. 

```bash
cd <path-to-crazyflie-firmware>

# Replace <path-to-Eigen> with path to Eigen
sudo docker run --rm -it -v $(pwd):/build -v <path-to-Eigen>:<path-to-Eigen> -w /build bitcraze/builder bash

# Inside the container:
make clean && make cf2_defconfig && make -j$(nproc)
```

## 9. Bring-up notes

- Check the `ga_stt.exec_us` log variable — it's the wall-clock time
  of `compute_control()` per call.
- `ga_stt.nan_guards` counts ticks where the controller output wasn't
  finite and was zeroed out as a safety fallback.
