package net.electroland.eio.devices.modbus;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import net.electroland.eio.Device;
import net.electroland.eio.InputChannel;
import net.electroland.eio.Value;
import net.electroland.eio.ValueSet;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;
import net.wimpi.modbus.ModbusException;
import net.wimpi.modbus.facade.ModbusTCPMaster;
import net.wimpi.modbus.procimg.InputRegister;

import org.apache.log4j.Logger;

public class ModBusTcpDevice extends Device {

    static Logger logger = Logger.getLogger(ModBusTcpDevice.class);

    private ModBusTcpInputChannel[][] channels; // ModBusTcpInputChannel[register][byte(0|1)]
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
        channels            = new ModBusTcpInputChannel[totalRegisters][2];

        // BIG or SMALL (no check to see if !"big" is actually "small")
        endianness         = params.getRequired("endianness").equalsIgnoreCase("big") ? 
                                ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN;
    }

    // TODO: this behaves badly. when connections are severed, it blocks. MUST FIX!
    @Override
    public ValueSet read() {

        ValueSet values = new ValueSet();

        // lazy init connection
        if (connection == null){
            connection = new ModbusTCPMaster(address, port);
            connection.setReconnecting(true); // hold persistent connection.
            try {
                connection.connect();

                // ModbusTCPMaster.connect() throws generic Exception. Fuck you Jamod.
            } catch (Exception e) {
                if (e instanceof java.net.ConnectException){
                    logger.error("connection was refused. retrying...");
                    // do not put a timeout in here. if someone is calling read and
                    // elu.sync synchronously, you'll be locking the display!
                }else{
                    e.printStackTrace();
                }
            }
        }

        if (connection != null){ // if connection (above) fails, it will be null
            try {

                // can still end up with null pointer exceptions here.
                InputRegister[] data = connection.readInputRegisters(startRef, totalRegisters);
                int registerIdx = 0;

                for (InputRegister register : data){

                    ByteBuffer bytes = ByteBuffer.allocate(2);
                    bytes.order(endianness);
                    for (byte b : register.toBytes()){
                        bytes.put(b);
                    }

                    // kludgy. won't scale to "bit1, bit2...bit8"
                    Eval e1 = evalChannel(bytes, registerIdx, 0); // first bypte
                    if (e1 != null){
                        values.put(e1.channel, e1.value);
                    }

                    Eval e2 = evalChannel(bytes, registerIdx, 1); // (optional) second byte
                    if (e2 != null){
                        values.put(e2.channel, e2.value);
                    }

                    registerIdx++;
                }

            } catch (ModbusException e) {
                e.printStackTrace();
            }
        }

        // will return an empty set if anything goes wrong.
        return values;
    }

    public Eval evalChannel(ByteBuffer bytes, int registerIdx, int byteIdx){
        ModBusTcpInputChannel channel = channels[registerIdx][byteIdx];
        if (channel == null){
            return null;
        }
        switch(channel.type){
            case BYTE:
                return new Eval(channel, new Value(bytes.get(byteIdx)));
            case SHORT:
                return new Eval(channel, new Value(bytes.getShort(0)));
            case UNINT:
                return new Eval(channel, new Value((int)bytes.getShort(0)));
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
        int whichByte = 0;
        int register = channelParams.getRequiredInt("register");

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