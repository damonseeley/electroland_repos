package net.electroland.skate.util;

/**
 * Handles all of the OSC message output and keeps a ticker to track player ID's.
 */

import java.awt.geom.Point2D;
import java.text.DecimalFormat;

import org.apache.log4j.Logger;


public class SpeakerLocCalc {

	private Point2D.Double audioListenerPos = new Point2D.Double(10,10);

	private static Logger logger = Logger.getLogger(SpeakerLocCalc.class);

	public SpeakerLocCalc(Point2D.Double listenPos) {

		audioListenerPos.x = listenPos.x;
		audioListenerPos.y = listenPos.y;
		logger.info("LISTENPOS = " + audioListenerPos.x + ", INCOMING = " + listenPos.x);

		Point2D.Double[] speakerLocs = new Point2D.Double[12];

		speakerLocs[0] = new Point2D.Double(1238.25,673.786);
		speakerLocs[1] = new Point2D.Double(1421.438,797.099);
		speakerLocs[2] = new Point2D.Double(1421.438,1025.967);
		speakerLocs[3] = new Point2D.Double(1421.438,1252.562);
		speakerLocs[4] = new Point2D.Double(1287.015,1385.495);
		speakerLocs[5] = new Point2D.Double(1081.91,1385.495);
		speakerLocs[6] = new Point2D.Double(880.508,1385.495);
		speakerLocs[7] = new Point2D.Double(753.075,1238.126);
		speakerLocs[8] = new Point2D.Double(753.075,963.622);
		speakerLocs[9] = new Point2D.Double(753.075,688.556);
		speakerLocs[10] = new Point2D.Double(821.988,513.007);
		speakerLocs[11] = new Point2D.Double(1074.04,513.007);

		for (int i=0; i < speakerLocs.length; i++){
			double az = roundTwoDecimals(computeAzimuth(audioListenerPos,speakerLocs[i]));
			double dist = roundTwoDecimals(computeDistanceInMeters(audioListenerPos,speakerLocs[i]));
			System.out.println("Speaker " + (i+1) + " azimuth:" + az + " distance:" + dist);
		}

	}

	double roundTwoDecimals(double d) {
		DecimalFormat twoDForm = new DecimalFormat("#.##");
		return Double.valueOf(twoDForm.format(d));
	}

	// 2D (always assumes reference is "above" the listener)
	public static double computeAzimuth(Point2D.Double listener, 
			Point2D.Double object){

		// object in front (0) or behind user (180)
		if (listener.x == object.x){
			if (listener.y == object.y)
				return 0;
			//				return Double.NaN; // object on top of user.
			else
				return listener.y < object.y ? 180 : 0;
		}


		double radians = Math.atan2(object.y - listener.y, object.x - listener.x);
		double degrees = 90 + (180/Math.PI) * radians; 
		return degrees > 0 ? degrees : 360 + degrees;
	}

	// 2D
	public static double computeDistance(Point2D.Double listener, Point2D.Double object)
	{
		return listener.distance(object);
	}

	// 2D
	public static double computeDistanceInMeters(Point2D.Double listener, Point2D.Double object)
	{
		return listener.distance(object)/100;
	}

	public static void main(String[] args){
		//$listenerX 1087.34 $listenerY 949.25
		Point2D.Double listenPos = new Point2D.Double(1087.34,949.25);
		new SpeakerLocCalc(listenPos);
	}


}