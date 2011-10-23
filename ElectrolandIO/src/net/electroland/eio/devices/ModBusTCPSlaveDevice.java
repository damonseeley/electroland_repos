package net.electroland.eio.devices;

import java.util.Collection;
import java.util.HashMap;

import net.electroland.eio.IOState;
import net.electroland.eio.IState;
import net.electroland.utils.OptionException;
import net.wimpi.modbus.ModbusException;
import net.wimpi.modbus.facade.ModbusTCPMaster;
import net.wimpi.modbus.procimg.InputRegister;
import net.wimpi.modbus.util.BitVector;

public class ModBusTCPSlaveDevice extends IODevice {

    protected String address;
    protected Collection <ModBusTCPSlaveDeviceRegister> registers;
    protected HashMap <String, Integer> portToRegisterBit = new HashMap<String, Integer>();
    protected HashMap <Integer, IOState> registerBitToState = new HashMap <Integer, IOState>();
 
    ModbusTCPMaster connection;
    
    @Override
    public void connect() 
    {
        connection = new ModbusTCPMaster(address);
        try {
            connection.connect();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void sendOutput() 
    {
        // TODO Auto-generated method stub
    }

    @Override
    public synchronized void readInput() 
    {
        for (ModBusTCPSlaveDeviceRegister r : registers){
            try {
                InputRegister[] data = connection.readInputRegisters(r.startRef, r.length);
                // for now, we're only doing the first word. 
                //  (need to update syntax to reference words)
                byte[] b = data[0].toBytes();

                BitVector bv = BitVector.createBitVector(b);

                for (int i=0; i < bv.size(); i++)
                {
                    IState state =  (IState)(registerBitToState.get(i + r.startRef));
                    if (state != null)
                    {
                        state.setState(bv.getBit(i));
                    }
                }

            } catch (ModbusException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void close()
    {
        connection.disconnect();
    }

    @Override
    public void patch(IOState state, String port)
    {
        Integer bit = portToRegisterBit.get(port);
        if (bit == null){
            throw new OptionException("Can't find port '" + port + '\'');
        }
        registerBitToState.put(bit, state);
    }

    public String toString()
    {
        StringBuffer sb = new StringBuffer("{portToRegisterBit=");
        sb.append(portToRegisterBit);
        sb.append(", address=");
        sb.append(address);
        sb.append(", registers=");
        sb.append(registers);
        sb.append(", registerBitToState=");
        sb.append(registerBitToState);
        sb.append('}');
        return sb.toString();
    }
}