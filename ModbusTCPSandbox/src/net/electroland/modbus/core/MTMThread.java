package net.electroland.modbus.core;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import net.wimpi.modbus.ModbusException;
import net.wimpi.modbus.facade.ModbusTCPMaster;
import net.wimpi.modbus.procimg.InputRegister;
import net.wimpi.modbus.util.BitVector;

import org.apache.log4j.Logger;

public class MTMThread  extends Thread {

	ModbusTCPMaster mtm;

	//Thread stuff
	public static boolean isRunning;
	private static float framerate;
	private static Timer timer;
	public static long curTime = System.currentTimeMillis(); //  time of frame start to aviod lots of calls to System.getcurentTime()
	public static long elapsedTime = -1; //  time between start of cur frame and last frame to avoid re calculating passage of time allover the place
	InputStreamReader isr;
	BufferedReader br;

	public SensorPanel sp;

	static Logger logger = Logger.getLogger(MTMThread.class);

	public boolean[] sensorStates = new boolean[8];
	public boolean[] sensorChanged = new boolean[8];
	private int[] sensorBits = new int[16];
	public double[] tripTimes = new double[sensorBits.length];
	public double[] tripTimesCalc = new double[sensorBits.length];

	public MTMThread(String ip, int fr, SensorPanel sp) {

		framerate = fr;
		this.sp = sp;

		mtm = new ModbusTCPMaster(ip);
		try {
			logger.info("Attempting to connect to IP: " + ip);
			mtm.connect();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		double startTime = System.currentTimeMillis();

		for (int i=0; i<sensorStates.length; i++) {
			sensorStates[i] = false;
			sensorChanged[i] = false;
		}
		for (int i=0; i<sensorBits.length; i++){
			sensorBits[i] = 0;
			tripTimes[i] = startTime;
		}
		
		sp.paintSensors(sensorStates, sensorChanged);

		/////////////// THREAD STUFF
		isRunning = true;
		timer = new Timer(framerate);
		start();

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

		while (isRunning) {
			try {
				InputRegister[] regs = mtm.readInputRegisters(0, 1);
				updateSensorStates(regs[0].toBytes());

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

			//Thread ops
			timer.block();
		}

	}

	BitVector bv = new BitVector(16);
	BitVector bvPrev = new BitVector(16);
	public int lastInputTripped = 0;

	private void updateSensorStates(byte[] bytes) {

		BitVector bv = BitVector.createBitVector(bytes);
		//System.out.println(bv);

		boolean anyChange = false;

		//update sensor states
		for (int i=0; i < sensorStates.length; i++) {
			if (bv.getBit(i+8) == true){
				sensorStates[i] = true;
			} else {
				sensorStates[i] = false;
			}
			if (bv.getBit(i+8) != bvPrev.getBit(i+8)){
				anyChange = true;
				sensorChanged[i] = true;
				tripTimes[i] = System.currentTimeMillis();
			}
		}

		bvPrev = bv.createBitVector(bv.getBytes());

		/*
		//calculate trip times as time elapsed
		double tempTime = System.currentTimeMillis();
		for (int i=0; i < tripTimes.length; i++) {
			tripTimesCalc[i] =  tempTime - tripTimes[i];
		}
		 */

		if (anyChange) {
			sp.paintSensors(sensorStates, sensorChanged);
			//System.out.println("Last input tripped = " + (lastInputTripped+1));
		}

		//reset
		for (int i=0; i<sensorChanged.length; i++) {
			sensorChanged[i] = false;
		}

	}


	private void printSensorStates() {
		for (int i=0; i < sensorStates.length; i++) {
			System.out.print(sensorStates[i] + " ");
		}
		System.out.println("");
	}

	private void printTripTimes() {
		for (int i=8; i < tripTimes.length; i++) {
			System.out.print(tripTimesCalc[i] + " - ");
		}
		System.out.println("---");
	}


	private void printSensorBits(BitVector bv) {
		for (int i=0; i < bv.size(); i++) {
			System.out.print(bv.getBit(i) + " ");
		}
		System.out.println(" ");
	}


	public void stopClean(){
		//logger.info(this + " is dying");
		stop();
		mtm.disconnect();
		isRunning = false;
	}

}
