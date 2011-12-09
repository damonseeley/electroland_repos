package net.electroland.eio.devices;

import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

public class RecordedModBusTCPSlaveDeviceFactory extends ModBusTCPSlaveDeviceFactory {

    @Override
    public IODevice createInstance(ParameterMap params) throws OptionException {
        // ionode.phoenix1 = $type phoenix4DI8D0 $ipaddress 192.168.1.61
        // create an instance of IODevice
        RecordedModBusTCPSlaveDevice node = new RecordedModBusTCPSlaveDevice();
        node.address = params.getRequired("ipaddress");
        node.filename = params.getRequired("playbackFile");
        node.portToRegisterBit = portToRegisterBit;
        node.registers = registers.values();

        return node;    }

}
