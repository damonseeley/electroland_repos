package net.electroland.eio.devices.modbus;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collection;

import net.electroland.eio.Device;
import net.electroland.eio.InputChannel;
import net.electroland.eio.Value;
import net.electroland.eio.ValueSet;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;
import net.wimpi.modbus.facade.ModbusTCPMaster;
import net.wimpi.modbus.procimg.InputRegister;

import org.apache.log4j.Logger;

public class ModBusTcpDevice extends Device {

    static Logger logger = Logger.getLogger(ModBusTcpDevice.class);

    private Collection<ModBusTcpInputChannel> channels;
    private String address;
    private int startRef, totalRegisters, port;
    private ByteOrder endianness;
    private ModbusTCPMaster connection;

    public ModBusTcpDevice(ParameterMap params) {
        super(params);

        // address of Inet4 ModBusTCP device we'll be reading from
        address             = params.getRequired("address");

        // port of Inet4 ModBusTCP device we'll be reading from. default is 502
        port                = params.getDefaultInt("port", 502);

        // first byte to start reading
        startRef            = params.getRequiredInt("startRef");

        // how many registeres to read
        totalRegisters      = params.getRequiredInt("totalRegisters");
        channels            = new ArrayList<ModBusTcpInputChannel>();

        // BIG or SMALL (no check to see if !"big" is actually "small")
        endianness         = params.getRequired("endianness").equalsIgnoreCase("big") ? 
                                ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN;
    }

    long retryTimeout = 500;
    @Override
    public ValueSet read() {

        ValueSet values = new ValueSet();

        try {

            // lazy init connection
            if (connection == null){
                connection = new ModbusTCPMaster(address, port);
                connection.setReconnecting(true); // hold persistent connection.
                connection.connect();
                retryTimeout = 500;
            }

            // read all registers & convert to bytes
            InputRegister[] registers = connection.readInputRegisters(startRef, totalRegisters);
            ByteBuffer[] registerBytes = new ByteBuffer[registers.length];

            for (int i = 0; i < registerBytes.length; i++){
                registerBytes[i] = toByteBuffer(registers[i], endianness);
            }

            // map registers to evaluated channel values
            for (ModBusTcpInputChannel channel : channels){
                Value value = eval(channel, registerBytes);
                values.put(channel, value);
            }
            return values;

        } catch (Exception e) { // ModbusTCPMaster.connect() throws generic Exception. Fuck you Jamod.

            logger.error(e);
            connection = null;

            try {
                logger.error("retry in " + retryTimeout + " millis.");
                Thread.sleep(retryTimeout);
                retryTimeout *= 2;

            } catch (InterruptedException e1) {
                e1.printStackTrace();
            }
        }

        // will return an empty set if anything goes wrong.
        return values;
    }

    private static ByteBuffer toByteBuffer(InputRegister register, ByteOrder endianness){
        ByteBuffer bb = ByteBuffer.allocate(2);
        bb.order(endianness);
        for (byte b : register.toBytes()){
            bb.put(b);
        }
        return bb;
    }


    public Value eval(ModBusTcpInputChannel channel, ByteBuffer[] registerBytes){

        // get the register
        ByteBuffer bytes = registerBytes[channel.registerIndex];

        // get the byte index
        switch(channel.type){
            case BYTE:
                return new Value(bytes.get(channel.byteIndex));
            case SHORT:
                return new Value(bytes.getShort(0));
            case UNINT:
                return new Value((int)bytes.getShort(0));
            default:
                return null;
        }
    }

    @Override
    public void write(ValueSet values) {
        // TODO Auto-generated method stub
    }

    @Override
    public InputChannel patch(ParameterMap channelParams) {

        logger.info("adding channel " + channelParams);

        // ichannel.ic2 = $device phoenix1 $register 1 $datatype SHORT
        ModBusTcpInputChannel newChannel = new ModBusTcpInputChannel();
        newChannel.byteIndex             = 0;
        newChannel.registerIndex         = channelParams.getRequiredInt("register");

        // TODO: need to handle BIT1 to BIT8 here.
        // set the datatype
        String type = channelParams.getRequired("datatype");
        if ("SHORT".equalsIgnoreCase(type)) {
            newChannel.type = ModBusTcpInputChannel.Type.SHORT;
        } else if ("UNINT".equalsIgnoreCase(type)) {
            newChannel.type = ModBusTcpInputChannel.Type.UNINT;
        } else if ("BYTE1".equalsIgnoreCase(type)) {
            newChannel.type = ModBusTcpInputChannel.Type.BYTE;
        } else if ("BYTE2".equalsIgnoreCase(type)) {
            newChannel.type = ModBusTcpInputChannel.Type.BYTE;
            newChannel.byteIndex = 1;
        } else {
            throw new OptionException("no ModBusTcp datatype '" + type + "'");
        }

        channels.add(newChannel);
        return newChannel;
    }

    @Override
    public void close() {
        connection.disconnect();
    }
}