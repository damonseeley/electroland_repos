package net.electroland.elvis.regions;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Vector;

import javax.swing.JFileChooser;
import javax.swing.JOptionPane;

public class GlobalRegionSnapshot implements Serializable {
	public double backgroundAdaptation;
	public double backgroundDiffThresh;
	public Vector<PolyRegion> regions;

	public GlobalRegionSnapshot(double adapt, double thresh, Vector<PolyRegion> r) {
		backgroundAdaptation = adapt;
		backgroundDiffThresh = thresh;
		regions = r;
	}

	public void save() {
		JFileChooser chooser = new JFileChooser();
		chooser.setDialogTitle("Save ElVis file...");
		if(chooser.showSaveDialog(null) == JFileChooser.APPROVE_OPTION) {
			File f =  chooser.getSelectedFile ();
			if(f.exists()) {
				int response = JOptionPane.showConfirmDialog (null,
						"Overwrite existing file?","Confirm Overwrite",
						JOptionPane.OK_CANCEL_OPTION,
						JOptionPane.QUESTION_MESSAGE);
				if (response == JOptionPane.CANCEL_OPTION) return ;
			}

			FileOutputStream fos;
			try {
				fos = new FileOutputStream(f);
				ObjectOutputStream oos = new ObjectOutputStream(fos);
				oos.writeObject(this);
				oos.close();
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}


		}
	}
	
	public static GlobalRegionSnapshot load(File f) {
		try {
			FileInputStream fis = new FileInputStream(f);
			ObjectInputStream ois = new ObjectInputStream(fis);
			GlobalRegionSnapshot grs = (GlobalRegionSnapshot)  ois.readObject();
			ois.close();
			for(PolyRegion p : grs.regions) {
				p.restoreTransients();
			}
			return grs;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	
	public static GlobalRegionSnapshot load() {
		JFileChooser chooser = new JFileChooser();
		chooser.setDialogTitle("Open ElVis file...");
		if(chooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
			File f = chooser.getSelectedFile();
			return load(f);
		}
		return null;
	}
}
