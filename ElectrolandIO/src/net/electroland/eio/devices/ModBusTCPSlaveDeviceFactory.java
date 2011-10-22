package net.electroland.eio.devices;

import java.util.HashMap;
import java.util.Map;

import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

public class ModBusTCPSlaveDeviceFactory extends IODeviceFactory {

    // maps port (by name) to regester start bit + offset bit (this is the actual port map)
    HashMap<String, Integer> portToRegisterBit = new HashMap<String, Integer>();
    // store registers
    HashMap<String, ModBusTCPSlaveDeviceRegister> registers = new HashMap<String, ModBusTCPSlaveDeviceRegister>();

    @Override
    public void prototypeDevice(Map<String, ParameterMap> props) throws OptionException{


        // get registers (name starts with register: kludgy)
        for (String s: props.keySet())
        {
            // for each register
            if (s.startsWith("register"))
            {
                // phoenix4DI8DO.register.1 = $startRef 0
                //   store the register start bit and length
                int startRef = props.get(s).getRequiredInt("startRef");
                int length = props.get(s).getRequiredInt("length");
                registers.put(s, new ModBusTCPSlaveDeviceRegister(startRef, length));
            }
        }
        // get patches
        // for each patch
        for (String s: props.keySet())
        {
            if (s.startsWith("patch"))
            {
                // phoenix4DI8DO.patch.0 = $register register.1 $bit 8 $port 0
                //   find the register, store the register + bit hashed by port
                // TODO: should check to see if register comes back null.
                String port = props.get(s).getRequired("port");
                int startRef = registers.get(props.get(s).getRequired("register")).startRef;
                int bit = props.get(s).getRequiredInt("bit");
                portToRegisterBit.put(port, startRef + bit);
            }
        }
    }

    @Override
    public IODevice createInstance(ParameterMap params) throws OptionException{
        // ionode.phoenix1 = $type phoenix4DI8D0 $ipaddress 192.168.1.61
        // create an instance of IODevice
        ModBusTCPSlaveDevice node = new ModBusTCPSlaveDevice();
        node.address = params.getRequired("ipaddress");
        node.portToRegisterBit = portToRegisterBit;
        node.registers = registers.values();

        return node;
    }
}