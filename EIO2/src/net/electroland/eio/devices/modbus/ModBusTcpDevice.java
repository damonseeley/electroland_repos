package net.electroland.eio.devices.modbus;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

import net.electroland.eio.devices.Device;
import net.electroland.eio.devices.InputChannel;
import net.electroland.eio.devices.OutputChannel;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;
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
    public Map<InputChannel, Object> read() {

        try{
            HashMap<InputChannel, Object> map = new HashMap<InputChannel, Object>();
            // lazy connet
            if (connection == null){
                connection = new ModbusTCPMaster(address);
                connection.setReconnecting(false);
                connection.connect();
            }

            InputRegister[] data;
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
            return map;
        } catch(Exception e){ // seriously, ModbusTCPMaster.connect() throws Exception? Fuck you Jamod.
            throw new ModBusTcpReadException(e);
        }
    }

    public Eval evalChannel(ByteBuffer bytes, int registerIdx, int byteIdx){
        ModBusTcpInputChannel channel = channels[registerIdx][byteIdx];
        switch(channel.type){
            case(ModBusTcpInputChannel.BYTE):
                return new Eval(channel, bytes.get(byteIdx));
            case(ModBusTcpInputChannel.SHORT):
                return new Eval(channel, bytes.getShort());
            case(ModBusTcpInputChannel.UNINT):
                return new Eval(channel, (int)bytes.getShort());
            default:
                return null;
        }
    }

    @Override
    public void write(Map<OutputChannel, Object> values) {
        // TODO Auto-generated method stub
    }

    @Override
    public InputChannel addInputChannel(ParameterMap channelParams) {

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
        return null;
    }

    class Eval {
        InputChannel channel;
        Object value;
        public Eval(InputChannel channel, Object value){
            this.channel = channel;
            this.value   = value;
        }
    }
}