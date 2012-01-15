package net.electroland.scSoundControl;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;
import java.util.Vector;

/**
 * This class exists as a convenience container for all the scsynth process startup parameters
 * 
 * Notes on the properties file settings:
 * If you see:
 * Exception in World_New: alloc failed
 * You need to either reduce max polyphony or increase realtime ram available.
 * Memory needs are a product of maxPolyphony and the number of audio output channels. 
 */

public class ScsynthLauncher {

	ProcessBuilder _builder;
	Process _scsynthProcess;
	SCSoundControl _sc;

	/* supercollider_synth  options:
	   -u <udp-port-number>    a port number 0-65535
	   -t <tcp-port-number>    a port number 0-65535
	   -c <number-of-control-bus-channels> (default 4096)
	   -a <number-of-audio-bus-channels>   (default 128)
	   -i <number-of-input-bus-channels>   (default 8)
	   -o <number-of-output-bus-channels>  (default 8)
	   -z <block-size>                     (default 64)
	   -Z <hardware-buffer-size>           (default 0)
	   -S <hardware-sample-rate>           (default 0)
	   -b <number-of-sample-buffers>       (default 1024)
	   -n <max-number-of-nodes>            (default 1024)
	   -d <max-number-of-synth-defs>       (default 1024)
	   -m <real-time-memory-size>          (default 8192)
	   -w <number-of-wire-buffers>         (default 64)
	   -r <number-of-random-seeds>         (default 64)
	   -D <load synthdefs? 1 or 0>         (default 1)
	   -R <publish to Rendezvous? 1 or 0>  (default 64)
	   -l <max-logins>                     (default -1073743192)
	          maximum number of named return addresses stored
	          also maximum number of tcp connections accepted
	   -p <session-password>
	          When using TCP, the session password must be the first command sent.
	          The default is no password.
	          UDP ports never require passwords, so for security use TCP.
	   -N <cmd-filename> <input-filename> <output-filename> <sample-rate> <header-format> <sample-format>
	   -I <input-streams-enabled>
	   -O <output-streams-enabled>
	   -M <server-mach-port-name> <reply-mach-port-name>
	   -H <hardware-device-name>
	   -v <verbosity>
	          0 is normal behaviour
	          -1 suppresses informational messages
	          -2 suppresses informational and many error messages
	 */

	//the relevant parameters:	
	//CAUTION when changing this array - it gets referenced via hardcoded subscript down below.
	private String[][] _scsynthParams = {
			{"SuperCollider_NumberOfInputChannels", "-i", "8"},
			{"SuperCollider_NumberOfOutputChannels", "-o", "8"},
			{"SuperCollider_RealTimeRamAllocation", "-m", "8192"},
			{"SuperCollider_UDPportNumber", "-u", "57110"}, 
			{"SuperCollider_LoadSynthDefs", "-D", "1"},
			{"SuperCollider_AudioHardwareDeviceName", "-H", ""},
			{"SuperCollider_SampleRate", "-S", "44100"}
	};
	//{"SuperCollider_AudioHardwareDeviceName", "-H", ""},
	//{"SuperCollider_InputStreamsEnabled", "-I", "00000000"},
	//{"SuperCollider_OutputStreamsEnabled", "-O", "11111111"},

	private String _execPath_string = "SuperCollider_Path";
	private String _execPath = "";//"/Applications/SuperCollider/scsyth"; //on a mac

	private String _maxPolyphony_string = "SCSoundControl_MaxPolyphony";
	private String _maxPolyphony = "64";

	private Properties _props;

	public ScsynthLauncher(SCSoundControl sc, Properties p) {
		_sc = sc;
		_props = p;
		_builder = new ProcessBuilder();
		//set instance variables from the properties object
		loadProperties();
		//then immediately commit their current values back to the properties object.
		//why? In case some values were undefined in the properties object.
		commitProperties();
	}

	/**
	 * Fire it up.
	 * Make sure launch properties are properly set!
	 */
	public void launch() {
		//don't launch if the path is empty
		if (_execPath.compareTo("") == 0) return;
		
		if (_scsynthProcess != null) killScsynth();

		/* Calculate values based on max polyphony:
		 * There are 2 system nodes (0 and 1) and a mothergroup node. Total 3.		
		*  Each voice is composed of 1 group, 1 ELplaybuf node & (N) ELenv nodes, where N = num output channels.
		*  For stereo buffers, two voices share one stereo ELplaybuf node and 1 group, but use 2N ELenv nodes.
		*  For the node math, we'll just treat every voice as a mono node, since that's slightly higher.
		*  So max Nodes = 3 + ((polyphony) * (2 + numOutputChannels))
		*  
		*  Each voice uses 1 audio bus to connect the output of the ELplaybuf node to all the ELenv nodes.
		*  The ELenv nodes then connect directly to the hardware output busses. 
		*  Plus there are input and output channel busses...
		*/
		
		//These should not have hardcoded subscripts. Should be searching the array for a string match:
		int maxNodes = 3 + (Integer.valueOf(_maxPolyphony) * (2 + Integer.valueOf(_scsynthParams[1][2])));
		int numAudioBusses = Integer.valueOf(_scsynthParams[0][2]) 
							+ Integer.valueOf(_scsynthParams[1][2]) 
							+ Integer.valueOf(_maxPolyphony);
		
		
		Vector<String> args = new Vector<String>();

		args.add(_execPath);

		for (int i=0; i<_scsynthParams.length; i++) {
			args.add(_scsynthParams[i][1]);
			args.add(_scsynthParams[i][2]);
		}
		
		args.add("-a");
		args.add(String.valueOf(numAudioBusses));
		args.add("-n");
		args.add(String.valueOf(maxNodes));

		//we use no control busses, so leave 'em off.
		args.add("-c");
		args.add("0");
		
		System.out.println("launching scsynth:");
		System.out.println(args);
		
		_builder.command(args);
		_builder.directory(new File(_execPath).getParentFile());

		try {
			_scsynthProcess = _builder.start();
		} catch (IOException e1) {
			e1.printStackTrace();
		}

	}

	/**
	 * Commit the current launch parameters to the SCSoundControl properties object. 
	 */
	private void commitProperties() {
		for (int i=0; i < _scsynthParams.length; i++) {
			_props.setProperty(_scsynthParams[i][0], _scsynthParams[i][2]);
		}
		_props.setProperty(_execPath_string, _execPath);
		_props.setProperty(_maxPolyphony_string, _maxPolyphony);
	}

	/**
	 * Set launch parameters based on contents of Properties object.
	 * If elements are not present, default values are used.
	 */
	private void loadProperties() {
		for (int i=0; i < _scsynthParams.length; i++) {
			_scsynthParams[i][2] = _props.getProperty(_scsynthParams[i][0], _scsynthParams[i][2]);
		}
		_execPath = _props.getProperty(_execPath_string, _execPath);
		_maxPolyphony = _props.getProperty(_maxPolyphony_string, "64");
	}


	/**
	 * If scsynth is running, kill it.
	 * This is a hard, system level kill of the process, not a quit message.
	 */
	public void killScsynth() {
		System.out.println("killing scsynth.");
		if (_scsynthProcess != null) _scsynthProcess.destroy();
	}
	
	/**
	 * Return the standard output of the scsynth process.
	 * Used to feed a UI element to display output.
	 * @return
	 */
	public InputStream getScsynthOutput() {
		if (_scsynthProcess == null) return null;
		else return _scsynthProcess.getInputStream();
	}
	
	public int getMaxPolyphony() {
		return Integer.valueOf(_maxPolyphony);
	}
}
