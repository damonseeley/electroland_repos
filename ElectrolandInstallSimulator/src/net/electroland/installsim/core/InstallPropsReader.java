package net.electroland.installsim.core;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Enumeration;
import java.util.Vector;

import net.electroland.coopLights.core.Light;


public class InstallPropsReader {
	protected String filename;
	protected BufferedReader reader;
	
	
	public InstallPropsReader(String filename) throws FileNotFoundException {
		this.filename = filename;
		reader = new BufferedReader(new FileReader(filename));		
	}
	
	public void setupLights() throws IOException {
		Vector<Light>lightVec = new Vector<Light>(); 
		int lineCnt = 0;
		while(reader.ready()) {
			String line  = reader.readLine();
			String[] els = line.split(",");
			if(els.length != 3) {
				System.err.println("Malformed light definition in line " + lineCnt + ":  " + line + " of file " + filename);
			} else {
				int id = Integer.parseInt(els[0]);
				float x = Float.parseFloat(els[1]);
				float y = Float.parseFloat(els[2]);
				lightVec.add(new Light(id,x,y));
			}
			lineCnt++;
			
		}
		reader.close();
		//InstallSimMain.lights = new Light[lightVec.size()];
		Enumeration<Light> e = lightVec.elements();
		while(e.hasMoreElements()) {
			Light l = e.nextElement();
			//System.out.println(l);
			//InstallSimMain.lights[l.id] = l;
		}
	}
}