package net.electroland.udpUtils;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.concurrent.TimeUnit;

import net.electroland.enteractive.utils.HexUtils;

import org.apache.log4j.Logger;


public class UDPParser extends Thread {

	// logger
	static Logger logger = Logger.getLogger(UDPParser.class);	
	
	UDPReceiver receiver;
	boolean isRunning = true;
	TCUtil tcUtils;

	public UDPParser(int port, TCUtil tcUtils) throws SocketException, UnknownHostException {
		receiver = new UDPReceiver(port);
		this.tcUtils = tcUtils;
	}

	public void parseMsg(String msg) {
		//logger.debug("msg: " + msg);
		byte[] line = HexUtils.hexToBytes(msg);
		int offset = (int)(line[line.length-3] & 0xFF) + (int)(line[line.length-2] & 0xFF);	// determines starting position of payload values
		//System.out.println(offset);
		if(tcUtils != null){
			if(line[1] == Byte.valueOf(tcUtils.tileProps.getProperty("TileUpdateCmd")).byteValue()){
				// direct response of light values
				lightValues(offset);
			} else if(line[1] == Byte.valueOf(tcUtils.tileProps.getProperty("TileFeedbackCmd")).byteValue()){
				// direct response of sensor states
				sensorValues(offset);
			} else if(line[1] == Byte.valueOf(tcUtils.tileProps.getProperty("TileOnOffCmd")).byteValue()){
				// direct response from tiles being powered on/off
				tilePowerState(offset);
			} else if(line[1] == Byte.valueOf(tcUtils.tileProps.getProperty("TileReportCmd")).byteValue()){
				// continuous sensor state reporting on change
				sensorValues(offset);
			} else if(line[1] == Byte.valueOf(tcUtils.tileProps.getProperty("TileOffsetCmd")).byteValue()){
				// TODO direct response from setting offset value
			} else if(line[1] == Byte.valueOf(tcUtils.tileProps.getProperty("TileMCResetCmd")).byteValue()){
				// TODO direct response from resetting microcontroller on tile controller
			} 
		} 
	}
	
	public void lightValues(int offset){
		
	}
	
	public void sensorValues(int offset){
		
	}
	
	public void tilePowerState(int offset){
		
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
	public static void main(String[] args) throws SocketException, UnknownHostException {
		UDPParser parser = new UDPParser(10011, null) ;
		parser.start();
	}

}
