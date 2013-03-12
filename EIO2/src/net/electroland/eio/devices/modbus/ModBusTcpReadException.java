package net.electroland.eio.devices.modbus;

@SuppressWarnings("serial")
public class ModBusTcpReadException extends RuntimeException {
    public ModBusTcpReadException(){
        super();
    }
    public ModBusTcpReadException(String message){
        super(message);
    }
    public ModBusTcpReadException(Exception e){
        super(e);
    }
}