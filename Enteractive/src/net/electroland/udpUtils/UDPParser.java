package net.electroland.udpUtils;

import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import net.electroland.enteractive.core.PersonTracker;
import net.electroland.enteractive.core.Tile;
import net.electroland.enteractive.utils.HexUtils;
import net.electroland.util.NoDataException;

import org.apache.log4j.Logger;


public class UDPParser extends Thread {

	// logger
	static Logger logger = Logger.getLogger(UDPParser.class);	
	
	UDPReceiver receiver;
	boolean isRunning = true;
	TCUtil tcUtils;
	PersonTracker personTracker;
	long lastTileCheck;
	int tileCheckDuration = 33;

	public UDPParser(int port, TCUtil tcUtils, PersonTracker personTracker) throws SocketException, UnknownHostException {
		receiver = new UDPReceiver(port);
		this.tcUtils = tcUtils;
		this.personTracker = personTracker;
		lastTileCheck = System.currentTimeMillis();
	}

	public void parseMsg(String msg) {
		//logger.debug("msg: " + msg);
		byte[] line = HexUtils.hexToBytes(msg);
		if((int)(line[0] & 0xFF) == 255 && (int)(line[line.length-1] & 0xFF) == 254){				// if it has start and end bytes...
			if(line.length >= 5){																	// packet should be at least 5 bytes long...
				byte[] data = new byte[line.length - 5];												// everything but start/end/offset/command bytes
				String commandByte = Integer.toHexString(line[1]);
				System.arraycopy(line, 2, data, 0, line.length-5);
				if(tcUtils != null){
					if(commandByte.equals(tcUtils.tileProps.getProperty("updateByte"))){				// direct response of light values
						int offset = (int)(line[line.length-3] & 0xFF) + (int)(line[line.length-2] & 0xFF);	// determines starting position of payload values
						lightValues(offset, data);
					} else if(commandByte.equals(tcUtils.tileProps.getProperty("feedbackByte"))){		// direct response of sensor states
						if(data.length == 8){
							int offset = (int)(line[line.length-3] & 0xFF) + (int)(line[line.length-2] & 0xFF);	// determines starting position of payload values
							sensorValues(offset, data);
						}
					} else if(commandByte.equals(tcUtils.tileProps.getProperty("powerByte"))){			// direct response from tiles being powered on/off
						int offset = (int)(line[line.length-3] & 0xFF) + (int)(line[line.length-2] & 0xFF);	// determines starting position of payload values
						tilePowerState(offset, data);
					} else if(commandByte.equals(tcUtils.tileProps.getProperty("reportByte"))){		// continuous sensor state reporting on change
						if(data.length == 8){
							int offset = (int)(line[line.length-3] & 0xFF) + (int)(line[line.length-2] & 0xFF);	// determines starting position of payload values
							sensorValues(offset, data);
						}
					} else if(commandByte.equals(tcUtils.tileProps.getProperty("offsetByte"))){
						// TODO direct response from setting offset value
					} else if(commandByte.equals(tcUtils.tileProps.getProperty("mcResetByte"))){
						// TODO direct response from resetting microcontroller on tile controller
					} 
				} 
			}
		}
	}
	
	public void lightValues(int offset, byte[] data){
		//logger.debug("offset: "+ offset + ", light values: " + HexUtils.bytesToHex(data, data.length));
		tcUtils.updateLights(offset, data);
	}
	
	public void sensorValues(int offset, byte[] data){
		//logger.debug("offset: "+ offset + ", sensor states: " + HexUtils.bytesToHex(data, data.length));
		tcUtils.updateSensors(offset, data);
		Map<Integer, Tile>stuckTiles = tcUtils.getStuckTiles();
		personTracker.updateSensors(offset, data, stuckTiles);
	}
	
	public void tilePowerState(int offset, byte[] data){
		//logger.debug("offset: "+ offset + ", tile power states: " + HexUtils.bytesToHex(data, data.length));
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
			
			
			/*
			double avg = 0;
			try {
				avg = personTracker.getModel().getAverage();
			} catch (NoDataException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			System.out.println("MODEL PEOPLE SIZE: " + (double)personTracker.getModel().getPeople().size() + " AND AVERAGE = " + avg);
			*/

			
			personTracker.updateAverage((double)personTracker.getModel().getPeople().size());
			
			if(System.currentTimeMillis() - lastTileCheck > tileCheckDuration){
				tcUtils.checkTileStates();
				lastTileCheck = System.currentTimeMillis();
			}
			
		}
	}




	//just for testing
//	public static void main(String[] args) throws SocketException, UnknownHostException {
//		UDPParser parser = new UDPParser(10011, null, null) ;
//		parser.start();
//	}

}
