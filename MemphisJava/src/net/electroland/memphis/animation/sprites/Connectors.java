package net.electroland.memphis.animation.sprites;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

import net.electroland.lighting.detector.animation.Raster;
import net.electroland.memphis.core.BridgeState;

public class Connectors extends Sprite implements SpriteListener {
	
	//private Connector[] connectors = new Connector[20];
	ConcurrentHashMap<Integer, Sprite> connectors;
	private int connectorID = 0;
	private int row = 0;	// 0-3, to represent row of lights
	private BridgeState state;
	private int connectDuration = 1000;	// connect in 1 second
	private int timeoutDuration = 5000;	// hold line for 5 seconds
	private int fadeDuration = 1000;	// fade out in one second

	public Connectors(int id, Raster raster, float x, float y, BridgeState state) {
		super(id, raster, x, y);
		connectors = new ConcurrentHashMap<Integer, Sprite>();
		this.state = state;
	}

	public void draw() {
		// check if state has a new pair
		int[] ltt = state.lastTwoTripped();
        if (ltt != null) { 
			//int posA = 0;
			//int posB = 5;
        	/*
			connectors[connectorID] = new Connector(connectorID, raster, x, row*6, ltt[0], ltt[1], connectDuration, timeoutDuration, fadeDuration);
        	connectors[connectorID].setColor(255,255,0);
			if(connectorID == 0){
				if(connectors[connectors.length - 1] != null){
					connectors[connectors.length - 1].fadeOutAndDie();
				}
			} else {
				connectors[connectorID - 1].fadeOutAndDie();
			}
			*/
        	if(Math.abs(ltt[0] - ltt[1]) > 3){
	        	Connector connector = new Connector(connectorID, raster, x, row*6, ltt[0], ltt[1], connectDuration, timeoutDuration, fadeDuration);
	        	connector.setColor(255, 255, 0);
	        	connector.addListener(this);
	        	connectors.put(connectorID++, connector);
	        	if(connectorID > 0){
	        		//((Connector)connectors.get(connectorID - 1)).fadeOutAndDie();
	        	}
				connectorID++;
				row++;
				if(row > 3){
					row = 0;
				}
        	}
			//if(connectorID == connectors.length){
				//connectorID = 0;
			//}
        }
        /*
        for(int i=0; i<connectors.length; i++){
        	if(connectors[i] != null){
        		connectors[i].draw();
        	}
        }
        */
        Iterator<Sprite> iter = connectors.values().iterator();
		while(iter.hasNext()){
			Sprite sprite = (Sprite)iter.next();
			sprite.draw();
		}
	}

	public void spriteComplete(Sprite sprite) {
		connectors.remove(sprite.getID());
	}

}
