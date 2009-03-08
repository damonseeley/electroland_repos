package net.electroland.udpUtils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Properties;

import net.electroland.enteractive.core.Tile;

public class TCUtil {

	Properties tileProps;
	
	public TCUtil(){
		try{
			tileProps = new Properties();
			tileProps.load(new FileInputStream(new File("depends//tile.properties")));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void turnOffTile(Tile tile){
		// TODO cycle power off
	}
	
	public void turnOnTile(Tile tile){
		// TODO cycle power on
	}
	
	public void turnOffAllTiles(){
		// TODO cycle power off
	}
	
	public void turnOnAllTiles(){
		// TODO cycle power on
	}
	
}
