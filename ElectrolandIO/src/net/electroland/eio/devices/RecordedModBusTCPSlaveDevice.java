package net.electroland.eio.devices;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collection;
import java.util.HashMap;

import net.electroland.eio.IOState;
import net.electroland.eio.IState;
import net.electroland.utils.OptionException;
import net.electroland.utils.Util;
import net.wimpi.modbus.util.BitVector;

import org.apache.log4j.Logger;

public class RecordedModBusTCPSlaveDevice extends IODevice {

    private static Logger logger = Logger.getLogger(RecordedModBusTCPSlaveDevice.class);
    protected String address;
    protected Collection <ModBusTCPSlaveDeviceRegister> registers;
    protected String filename;
    protected BufferedReader reader;
    protected HashMap <String, Integer> portToRegisterBit = new HashMap<String, Integer>();
    protected HashMap <Integer, IOState> registerBitToState = new HashMap <Integer, IOState>();

    @Override
    public void connect() {
        reader = new BufferedReader(
                new InputStreamReader(
                        new Util().getClass().getClassLoader().getResourceAsStream(filename)));
    }

    @Override
    public void sendOutput() {
        // DO NOTHING
    }

    @Override
    public void readInput() {
        for (ModBusTCPSlaveDeviceRegister r : registers){

            // read a line
            // 192.168.247.23:00000000 00000000
            String[] data;
            try {
                data = reader.readLine().split(":");

                data[1] = data[1].replace(" ", "");
                if (address.equals(data[0])){
                    //System.out.println(data[0] + " : " + data[1]);
                    boolean[] bits = toBits(data[1]);
                    for (int i=0; i < bits.length; i++)
                    {
                        IState state =  (IState)(registerBitToState.get(i + r.startRef));
                        if (state != null)
                        {
                            state.setState(bits[i]);
                        }
                    }

                }
                } catch (IOException e) {
                    e.printStackTrace();
                }
        }
    }

    private static boolean[] toBits(String s){
        boolean[] bits = new boolean[s.length()];
        for (int i = 0; i < s.length(); i++)
        {
            bits[i] = s.charAt(i) == '1';
        }
        return bits;
    }
    
    @Override
    public void close() {
        // TODO Auto-generated method stub

    }

    @Override
    public void patch(IOState state, String port) {
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
