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
        logger.debug("attempting to connect to " + address);
        try {
            connection.connect();
        } catch (Exception e) {
            e.printStackTrace();
        }
        logger.debug("connected to " + address);
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
                StringBuffer sb = new StringBuffer(address).append(':');    // for debug
                for (int i=0; i < bv.size(); i++)
                {
                    sb.append(bv.getBit(i) ? "1" : "0");                    // for debug
                    if ((i+1) % 8 == 0){                                    // for debug
                        sb.append(' ');                                     // for debug
                    }                                                       // for debug
                    IState state =  (IState)(registerBitToState.get(i + r.startRef));
                    if (state != null)
                    {
                        state.setState(bv.getBit(i));
                    }
                }
                logger.debug(sb);                                           // for debug

            } catch (ModbusException e) {
                e.printStackTrace();
                connect();
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