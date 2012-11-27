package net.electroland.eio.devices.modbus;

import java.util.Map;

import net.electroland.eio.Value;
import net.electroland.eio.devices.Channel;
import net.electroland.eio.devices.Device;
import net.electroland.utils.ParameterMap;

public class ModBusTcpDevice extends Device {

    public ModBusTcpDevice(ParameterMap params) {
        super(params);
        // TODO Auto-generated constructor stub
    }

    @Override
    public Map<Channel, Value> read() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void write(Map<Channel, Value> values) {
        // TODO Auto-generated method stub
        
    }
}