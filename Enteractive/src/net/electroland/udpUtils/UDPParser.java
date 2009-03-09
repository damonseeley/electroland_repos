package net.electroland.udpUtils;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.concurrent.TimeUnit;

import net.electroland.enteractive.core.PersonTracker;
import net.electroland.enteractive.utils.HexUtils;

import org.apache.log4j.Logger;


public class UDPParser extends Thread {

	// logger
	static Logger logger = Logger.getLogger(UDPParser.class);	
	
	UDPReceiver receiver;
	boolean isRunning = true;
	TCUtil tcUtils;
	PersonTracker personTracker;

	public UDPParser(int port, TCUtil tcUtils, PersonTracker personTracker) throws SocketException, UnknownHostException {
		receiver = new UDPReceiver(port);
		this.tcUtils = tcUtils;
		this.personTracker = personTracker;
	}

	public void parseMsg(String msg) {
		//logger.debug("msg: " + msg);
		byte[] line = HexUtils.hexToBytes(msg);
		int offset = (int)(line[line.length-3] & 0xFF) + (int)(line[line.length-2] & 0xFF);	// determines starting position of payload values
		byte[] data = new byte[line.length - 5];												// everything but start/end/offset/command bytes
		String commandByte = "0x"+Integer.toHexString(line[1]);
		System.arraycopy(line, 2, data, 0, line.length-5);
		//System.out.println(offset);
		if(tcUtils != null){
			if(commandByte.equals(tcUtils.tileProps.getProperty("TileUpdateCmd"))){				// direct response of light values
				lightValues(offset, data);
			} else if(commandByte.equals(tcUtils.tileProps.getProperty("TileFeedbackCmd"))){	// direct response of sensor states
				sensorValues(offset, data);
			} else if(commandByte.equals(tcUtils.tileProps.getProperty("TileOnOffCmd"))){		// direct response from tiles being powered on/off
				tilePowerState(offset, data);
			} else if(commandByte.equals(tcUtils.tileProps.getProperty("TileReportCmd"))){		// continuous sensor state reporting on change
				sensorValues(offset, data);
			} else if(commandByte.equals(tcUtils.tileProps.getProperty("TileOffsetCmd"))){
				// TODO direct response from setting offset value
			} else if(commandByte.equals(tcUtils.tileProps.getProperty("TileMCResetCmd"))){
				// TODO direct response from resetting microcontroller on tile controller
			} 
		} 
	}
	
	public void lightValues(int offset, byte[] data){
		logger.debug("offset: "+ offset + ", light values: " + HexUtils.bytesToHex(data, data.length));
	}
	
	public void sensorValues(int offset, byte[] data){
		logger.debug("offset: "+ offset + ", sensor states: " + HexUtils.bytesToHex(data, data.length));
	}
	
	public void tilePowerState(int offset, byte[] data){
		logger.debug("offset: "+ offset + ", tile power states: " + HexUtils.bytesToHex(data, data.length));
	}

	public void stopRunning() {
		isRunning = false;
		receiver.stopRunning();
	}

	public void run() {
		receiver.start();
		while (isRunning) {
			try {
				String msg = receiver.msgQueue.poll(2000, TimeUnit.MILLISECONDS);
				if (msg != null) { // make sure didn't time out
					parseMsg(msg);
				}
			} catch (InterruptedException e) {
				logger.error(e.getMessage(), e);
			}
		}
	}




	//just for testing
//	public static void main(String[] args) throws SocketException, UnknownHostException {
//		UDPParser parser = new UDPParser(10011, null, null) ;
//		parser.start();
//	}

}
