package net.electroland.scSoundControl;

public class SoundNode {

	//always need a reference to the mother ship:
	SCSoundControl _sc;

	//keep track of various super collider IDs:
	protected int _bus;
	protected int _group;
	protected int _playbufID;
	protected int _buffer;

	//_envelopeNodeIDs values correspond directly to the audio channels stored in _outputChannels.
	protected int[][] _envelopeNodeIDs;

	
	//properties:
	//this is the number of channels in the playback buffer. Valid values are 1 and 2 (currently), corresponding to mono and stereo sounds.
	protected int _numBufferChannels;
	
	//_outputChannels[playBufferChannel][i] stores an audio outputChannel index
	protected int[][] _outputChannels; 

	//_amplitude values correspond directly to the audio channels stored in _outputChannels.
	protected float[][] _amplitude;
	

	protected boolean _looping;
	protected float _rate;
	protected boolean _alive; //is this node alive and playing on the server?
	
	/**
	 * 
	 * @param sc: reference to master SCSoundControl Object
	 * @param bufferNumber: ID of buffer to playback
	 * @param numBufferChannels: how many channels does the playback buffer have? (max 2)
	 * @param doLoop: boolean. True = loop, False = play then stop and die.
	 * @param outputChannels: 2D array of output channel indeces. For mono playback buffers, only use outputChannels[0].
	 * @param amplitude: 2D array of playback amplitudes. amplitude[i].length should equal outputChannels[i].length
	 * @param playbackRate: scale playback speed.
	 */
	public SoundNode(SCSoundControl sc, int bufferNumber, int numBufferChannels, boolean doLoop, int[][] outputChannels, float[][] amplitude, float playbackRate) {
		_sc = sc;
		_alive = true;
		_looping = doLoop;
		_numBufferChannels = numBufferChannels;
		_rate = playbackRate;
		_buffer = bufferNumber;

		//make a copy of the outputchannels and amplitude arrays
		//also initialize array for envelope soundNode IDs
		_outputChannels = new int[outputChannels.length][];
		_amplitude = new float[amplitude.length][];
		_envelopeNodeIDs = new int[outputChannels.length][];
		for (int i = 0; i < outputChannels.length; i ++ ) {
			_outputChannels[i] = new int[outputChannels[i].length];
			System.arraycopy(outputChannels[i], 0, _outputChannels[i], 0, outputChannels[i].length);

			_amplitude[i] = new float[amplitude[i].length];
			System.arraycopy(amplitude[i], 0, _amplitude[i], 0, amplitude[i].length);

			_envelopeNodeIDs[i] = new int[outputChannels[i].length];
		}
		
		_group = _sc.createGroup();
		_bus = _sc.allocateConsecutiveBusses(_numBufferChannels);

		//create a playbuffer
		_playbufID = _sc.getNewNodeID();
		if (_numBufferChannels == 1) {
			_sc.createPlayBuf(_playbufID, _group, _buffer, _bus, 1f, _rate, _looping);
		}
		else if (_numBufferChannels == 2) {
			_sc.createStereoPlayBuf(_playbufID, _group, _buffer, _bus, 1f, _rate, _looping);
		}
		else {
			_sc.debugPrintln("SoundNode: buffers over 2 channels (stereo sound files) are not supported");
		}


		//create an env for each specified output channel
		for (int i = 0; i < _outputChannels.length; i ++ ) {
			for (int j = 0; j < _outputChannels[i].length; j++) {
				_envelopeNodeIDs[i][j] = _sc.getNewNodeID();
				_sc.createEnvelope(_envelopeNodeIDs[i][j], _group, _bus + i, _outputChannels[i][j], _amplitude[i][j]);
			}			
		}
		
	}

	
	private int getNodeIdIndex(int bufferChannel, int outputChannel) {
		for (int i = 0; i < _outputChannels[bufferChannel].length; i++) {
			if (_outputChannels[bufferChannel][i] == outputChannel) return i;
		}
		return -1;
	}
	
	//multi channel version
	public void setAmplitude(int bufferChannel, int outputChannel, float level) {
		int idIndex = getNodeIdIndex(bufferChannel, outputChannel);
		
		if (idIndex < 0) {
			System.err.println("SoundNode::setAmplitude() : outputChannel value of " + outputChannel + " is invalid. Did you specify this as an output channel when you created this SoundNode?");
			return;
		}

		_amplitude[bufferChannel][idIndex] = level;
		_sc.setProperty(_envelopeNodeIDs[bufferChannel][idIndex], "ampScale", level);
	}
	
	//mono version
	public void setAmplitude(int channel, float level) {
		setAmplitude(0, channel, level);
	}

	//multi channel version
	public void setAmplitudes(int bufferChannel, int outputChannel[], float level[]) {
		for (int i = 0; i < Math.min(outputChannel.length, level.length); i++) {
			setAmplitude(bufferChannel, outputChannel[i], level[i]);
		}
	}
	
	//mono version
	public void setAmplitudes(int[] channel, float[] level) {
		setAmplitudes(0, channel, level);
	}
	
	public void setPlaybackRate(float playbackRate) {
		_rate = playbackRate;
		_sc.setProperty(_playbufID, "playbackRate", playbackRate);		
	}
	
	public void die() {
		_sc.freeNode(_group);
	}
	
	public void cleanup() {
		for (int i = 0; i < _numBufferChannels; i++) _sc.freeBus(_bus + i);		
	}

	//accessor and mutator methods:
	public int get_busID() {
		return _bus;
	}


	public boolean is_looping() {
		return _looping;
	}


	public void set_looping(boolean looping) {
		_looping = looping;
	}


	public boolean isAlive() {
		return _alive;
	}


	public void setAlive(boolean alive) {
		this._alive = alive;
	}

	public int getGroup() {
		return _group;
	}
}
