package net.electroland.eio.devices;

import java.util.Collection;
import java.util.HashMap;

import net.electroland.eio.IOState;
import net.electroland.eio.IState;
import net.wimpi.modbus.ModbusException;
import net.wimpi.modbus.facade.ModbusTCPMaster;
import net.wimpi.modbus.procimg.InputRegister;
import net.wimpi.modbus.util.BitVector;

import org.apache.log4j.Logger;

public class ModBusTCPSlaveDevice extends IODevice {

    private static Logger logger = Logger.getLogger(ModBusTCPSlaveDevice.class);
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
            // TODO Auto-generated catch block
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
//                logger.debug(address + ":\t bits 0 to " + (bv.size()-1) + ":\t " + bv.toString());

                for (int i=0; i < bv.size(); i++)
                {
                   IState state =  (IState)(registerBitToState.get(i + r.startRef));
                    if (state != null)
                    {
//                        System.out.println("setting " + (i + r.startRef) + " to " + bv.getBit(i));
                        state.setState(bv.getBit(i));
                    }else{
//                        System.out.println("failed " + (i + r.startRef) + " to " + bv.getBit(i));
                    }
                }
                // for each bit, see if a state is listening on it (+startRef)

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
        registerBitToState.put(portToRegisterBit.get(port), state);
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