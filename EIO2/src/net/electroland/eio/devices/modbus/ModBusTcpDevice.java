package net.electroland.eio.devices.modbus;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

import net.electroland.eio.Device;
import net.electroland.eio.InputChannel;
import net.electroland.eio.OutputChannel;
import net.electroland.eio.Value;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;
import net.wimpi.modbus.ModbusException;
import net.wimpi.modbus.facade.ModbusTCPMaster;
import net.wimpi.modbus.procimg.InputRegister;

public class ModBusTcpDevice extends Device {

    private ModBusTcpInputChannel[][] channels; // ModBusTcpInputChannel[register][byte(0|1)]
    private String address;
    private int startRef, totalRegisters;
    private ByteOrder endianness;
    private ModbusTCPMaster connection;

    public ModBusTcpDevice(ParameterMap params) {
        super(params);

        // address of Inet4 ModBusTCP device we'll be reading from
        address             = params.getRequired("address");

        // first byte to start reading
        startRef            = params.getRequiredInt("startRef");

        // how many registeres to read
        totalRegisters      = params.getRequiredInt("totalRegisters");
        channels            = new ModBusTcpInputChannel[totalRegisters][2];

        // BIG or SMALL (no check to see if !"big" is actually "small")
        endianness         = params.getRequired("endianness").equalsIgnoreCase("big") ? 
                                ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN;
    }

    @Override
    public Map<InputChannel, Value> read() {

        HashMap<InputChannel, Value> map = new HashMap<InputChannel, Value>();

        // lazy init connection
        if (connection == null){
            connection = new ModbusTCPMaster(address);
            connection.setReconnecting(false); // hold persistent connection.
            try {
                connection.connect();
                // ModbusTCPMaster.connect() throws generic Exception. Fuck you Jamod.
            } catch (Exception e) {
                throw new ModBusTcpReadException(e);
            }
        }

        InputRegister[] data;

        try {
            data = connection.readInputRegisters(startRef, totalRegisters);
            int registerIdx = 0;

            for (InputRegister register : data){

                ByteBuffer bytes = ByteBuffer.allocate(2);
                bytes.order(endianness);
                for (byte b : register.toBytes()){
                    bytes.put(b);
                }

                Eval e1 = evalChannel(bytes, registerIdx, 0); // first bypte
                if (e1 != null){
                    map.put(e1.channel, e1.value);
                }

                Eval e2 = evalChannel(bytes, registerIdx, 1); // (optional) second byte
                if (e2 != null){
                    map.put(e2.channel, e2.value);
                }

                registerIdx++;
            }

        } catch (ModbusException e) {
            // TODO: what's appropriate here? return a map of all suspect? all zeros?
            // retry?
        }

        return map;
    }

    public Eval evalChannel(ByteBuffer bytes, int registerIdx, int byteIdx){
        ModBusTcpInputChannel channel = channels[registerIdx][byteIdx];
        if (channel == null){
            return null;
        }
        switch(channel.type){
            case(ModBusTcpInputChannel.BYTE):
                return new Eval(channel, new Value(bytes.get(byteIdx)));
            case(ModBusTcpInputChannel.SHORT):
                return new Eval(channel, new Value(bytes.getShort(0)));
            case(ModBusTcpInputChannel.UNINT):
                return new Eval(channel, new Value((int)bytes.getShort(0)));
            default:
                return null;
        }
    }

    @Override
    public void write(Map<OutputChannel, Value> values) {
        // TODO Auto-generated method stub
    }

    @Override
    public InputChannel addInputChannel(ParameterMap channelParams) {

        System.out.println("adding channel " + channelParams);

        // ichannel.ic2 = $device phoenix1 $register 1 $datatype SHORT
        ModBusTcpInputChannel newChannel = new ModBusTcpInputChannel();
        int whichByte = 0;
        int register = channelParams.getRequiredInt("register");

        // set the datatype
        String type = channelParams.getRequired("datatype");
        if ("SHORT".equalsIgnoreCase(type)) {
            newChannel.type = ModBusTcpInputChannel.SHORT;
        } else if ("UNINT".equalsIgnoreCase(type)) {
            newChannel.type = ModBusTcpInputChannel.UNINT;
        } else if ("BYTE1".equalsIgnoreCase(type)) {
            newChannel.type = ModBusTcpInputChannel.BYTE;
        } else if ("BYTE2".equalsIgnoreCase(type)) {
            newChannel.type = ModBusTcpInputChannel.BYTE;
            whichByte = 1;
        } else {
            throw new OptionException("no ModBusTcp datatype '" + type + "'");
        }

        channels[register][whichByte] = newChannel;
        return newChannel;
    }

    class Eval {
        InputChannel channel;
        Value value;
        public Eval(InputChannel channel, Value value){
            this.channel = channel;
            this.value   = value;
        }
    }

    @Override
    public void close() {
        connection.disconnect();
    }
}