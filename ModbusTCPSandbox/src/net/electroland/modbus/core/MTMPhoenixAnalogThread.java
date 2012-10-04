package net.electroland.modbus.core;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import net.wimpi.modbus.ModbusException;
import net.wimpi.modbus.facade.ModbusTCPMaster;
import net.wimpi.modbus.procimg.InputRegister;
import net.wimpi.modbus.util.BitVector;

import org.apache.log4j.Logger;

public class MTMPhoenixAnalogThread  extends Thread {

	ModbusTCPMaster mtm;

	//Thread stuff
	public static boolean isRunning;
	private static float framerate;
	private static Timer timer;
	public static long curTime = System.currentTimeMillis(); //  time of frame start to aviod lots of calls to System.getcurentTime()
	public static long elapsedTime = -1; //  time between start of cur frame and last frame to avoid re calculating passage of time allover the place
	InputStreamReader isr;
	BufferedReader br;

	public String ip;
	
	public int regOffset;

	public SensorPanelAnalog sp;

	static Logger logger = Logger.getLogger(MTMPhoenixAnalogThread.class);

	public boolean[] sensorChanged = new boolean[8];
	private int[] sensorBits = new int[16];
	public double[] tripTimes = new double[sensorBits.length];
	public double[] tripTimesCalc = new double[sensorBits.length];

	public MTMPhoenixAnalogThread(String ip, int fr, SensorPanelAnalog sp) {

		framerate = fr;
		this.sp = sp;
		this.ip = ip;
		
		regOffset = 192; // the specific offset for PhoenixBusCoupler analog inputs

		mtm = new ModbusTCPMaster(ip);

		double startTime = System.currentTimeMillis();

		/////////////// THREAD STUFF
		isRunning = false;
		timer = new Timer(framerate);	

		start();
		//connectMTM();

	}

	public void connectMTM(String ip) {
		sp.drawBlank();
		this.ip = ip;
		mtm.disconnect();
		mtm = new ModbusTCPMaster(ip);
		try {
			logger.info("Attempting to connect to IP: " + ip);
			mtm.connect();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		isRunning = true;
	}

	public void disconnectMTM() {
		try {
			logger.info("Disconnecting");
			mtm.disconnect();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		sp.drawBlank();
		isRunning = false;
	}


	private double lastframe;
	private double execTime;
	private int cycle = 0;
	private int reportFreq = 100;
	private double[] execAvg = new double[reportFreq];

	public void run() {
		timer.start();
		curTime = System.currentTimeMillis();
		lastframe = curTime;

		while (true) {
			if (isRunning) {
				try {
					InputRegister[] regs = mtm.readInputRegisters(regOffset, 1);
	                InputRegister[] regs2 = mtm.readInputRegisters(regOffset+1, 1);

					//logger.info(regs[0].toBytes()[0] + "  " + regs2[0].toBytes()[0]);
					updateSensorStates(regs[0].toBytes()[0],regs2[0].toBytes()[0]);

					if (cycle >= reportFreq){
						double rateAvg = 0.0;
						for (int i=0; i< execAvg.length; i++){
							rateAvg += execAvg[i];
						}
						rateAvg  = rateAvg/execAvg.length;
						// for determining frame execution time
						//System.out.println("Frame Exec avg " + rateAvg + " ms");
						cycle = 0;
					}
				} catch (ModbusException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}

				execAvg[cycle] = System.currentTimeMillis() - lastframe;
				lastframe = System.currentTimeMillis();
				cycle++;
			}
			//Thread ops
			timer.block();
		}

	}

	BitVector bv = new BitVector(16);
	BitVector bvPrev = new BitVector(16);
	public int lastInputTripped = 0;

	private void updateSensorStates(byte v1, byte v2) {

			sp.paint2SensorsAnalog(v1,v2);

	}


	
	private void printTripTimes() {
		for (int i=8; i < tripTimes.length; i++) {
			System.out.print(tripTimesCalc[i] + " - ");
		}
		System.out.println("---");
	}




	public void stopClean(){
		//logger.info(this + " is dying");
		stop();
		mtm.disconnect();
		isRunning = false;
	}

}
