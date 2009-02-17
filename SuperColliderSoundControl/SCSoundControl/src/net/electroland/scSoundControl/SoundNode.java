package net.electroland.scSoundControl;

public class SoundNode {

	//always need a reference to the mother ship:
	SCSoundControl _sc;

	//keep track of various super collider IDs:
	protected int _bus;
	protected int _group;
	protected int _playbufID;
	protected int[][] _envelopeNodeIDs;
	protected int _buffer;

	//properties:
	protected int _numBufferChannels;
	protected int _numOutputChannels;
	protected boolean _looping;
	protected float[][] _amplitude;
	protected float _rate;
	protected boolean _alive; //is this node alive and playing on the server?
	
	public SoundNode(SCSoundControl sc, int bufferNumber, int numBufferChannels, boolean doLoop, int numOutputChannels, float[][] amplitude, float playbackRate) {
		_sc = sc;
		_alive = true;
		_looping = doLoop;
		_numBufferChannels = numBufferChannels;
		_numOutputChannels = numOutputChannels;
		_rate = playbackRate;
		_buffer = bufferNumber;
		
		_amplitude = new float[_numBufferChannels][_numOutputChannels];
		_envelopeNodeIDs = new int[_numBufferChannels][_numOutputChannels];

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
			_sc.debugPrintln("SoundNode: buffers over 2 channels (stereo sound files) not supported");
		}

		//hang on to amplitudes
		for (int i = 0; i < _numBufferChannels; i++) {
			for (int j = 0; j < _numOutputChannels; j++) {
				_amplitude[i][j] = amplitude[i][j];
			}
		}

		//create an env for each out channel
		for (int i = 0; i < _numBufferChannels; i++) {
			for (int j = 0; j < _numOutputChannels; j++) {
				_envelopeNodeIDs[i][j] = _sc.getNewNodeID();
				_sc.createEnvelope(_envelopeNodeIDs[i][j], _group, _bus + i, j, _amplitude[i][j]);
			}
		}
		
	}
		
	//multi channel version
	public void setAmplitude(int bufferChannel, int outputChannel, float level) {
		if (outputChannel < 0 || outputChannel > _numOutputChannels) System.err.println("SoundNode::setAmplitude() : channel value of " + outputChannel + " is outside valid range.");
		_amplitude[bufferChannel][outputChannel] = level;
		_sc.setProperty(_envelopeNodeIDs[bufferChannel][outputChannel], "ampScale", level);
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

	public int get_numChannels() {
		return _numOutputChannels;
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
