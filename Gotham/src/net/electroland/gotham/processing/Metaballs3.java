package net.electroland.gotham.processing;

import java.awt.Point;
import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.List;

import controlP5.ControlEvent;
import controlP5.ControlListener;

import net.electroland.elvis.net.GridData;
import net.electroland.gotham.processing.assets.Color;
import net.electroland.gotham.processing.assets.FastBlur;
import net.electroland.gotham.processing.assets.MetaballsGUI;
import processing.core.PVector;

@SuppressWarnings("serial")
public class Metaballs3 extends GothamPApplet {

    public static final int NONE                    = 0;
    public static final int MOUSE                   = 1;
    public static final int GRID                    = 2;
    public static final int GRID_DEFAULT            = 0;
    public static final int GRID_EDIT_TOPLEFT       = 1;
    public static final int GRID_EDIT_DIM           = 2;
    public static final int GRID_EDIT_INSET         = 3;
    public static final int GRID_EDIT_CANVAS_DIM    = 4;

    private int mode = GRID; // set the reaction mode
    private int presenceMode = GRID_DEFAULT;

    public static final float FRICTION            = .999f;  // higher = less .999
    public static final float MAX_VEL             = 1.0f;  // base velocity with no interaction or tracks higher = faster was 0.75
    // push vars
    public static final float MAX_RUN_VEL         = 1000.0f;  // max velocity when mouse is down or presence is felt.  //30
    //public static final float REPELL_FORCE        = 1000; // repell force of mouse or track (higher = more)
    //public static final int   RUN_TOLERANCE       = 30;
    // ball group props
    public static final float BALL_REPELL_FORCE   = 20;      // group to group repell force (higher = more)
    //public static final float COHESION            = .005f;   // higher = more was .005 .01 monday
    // ball scale
    public static final float ELLIPSE_SCALE       = 5.0f;   // percent 2.0-2.4 then 3.5 for most testting
    public static final int ELLIPSE_ALPHA         = 150;   // value/255
    // presence scale
    public static final float DILATE              = 6.0f;

    // don't touch:
    public static final float MIN_VEL             = .75f;   // higher = faster

    private List <MetaballGroup>groups;

    private GridData gridData;
    // specifies the section of the incoming grid that we want to subset
    private Rectangle grid = new Rectangle(2, 4, 68, 21);
    // specifies how the grid will be translated and scaled to the canvas
    private Rectangle gridOnCanvas = new Rectangle(80, 70, 0, 0);

    MetaballsGUI gui;


    @Override
    public void setup(){

        gui = new MetaballsGUI(this);

        // groups of balls
        groups = new ArrayList<MetaballGroup>();

        int redOrgRoam = 40; //was 100
        int purpRoam = 0; //was 0

        MetaballGroup red = new MetaballGroup(new Rectangle(-redOrgRoam, -redOrgRoam, this.getSyncArea().width + redOrgRoam, this.getSyncArea().height +redOrgRoam), gui.getColor1());
        gui.r1.addListener(red);
        gui.g1.addListener(red);
        gui.b1.addListener(red);
        groups.add(red);
        red.add(new Metaball(75 * ELLIPSE_SCALE));
        red.add(new Metaball(80 * ELLIPSE_SCALE));
        red.add(new Metaball(100 * ELLIPSE_SCALE));
        red.add(new Metaball(75 * ELLIPSE_SCALE));
        red.add(new Metaball(80 * ELLIPSE_SCALE));
        red.add(new Metaball(100 * ELLIPSE_SCALE));

        MetaballGroup orange = new MetaballGroup(new Rectangle(-redOrgRoam, -redOrgRoam, this.getSyncArea().width + redOrgRoam, this.getSyncArea().height +redOrgRoam), gui.getColor2());
        gui.r2.addListener(orange);
        gui.g2.addListener(orange);
        gui.b2.addListener(orange);
        groups.add(orange);
        orange.add(new Metaball(70 * ELLIPSE_SCALE));
        orange.add(new Metaball(80 * ELLIPSE_SCALE));
        orange.add(new Metaball(90 * ELLIPSE_SCALE));
        orange.add(new Metaball(70 * ELLIPSE_SCALE));
        orange.add(new Metaball(80 * ELLIPSE_SCALE));
        orange.add(new Metaball(90 * ELLIPSE_SCALE));

        MetaballGroup purple = new MetaballGroup(new Rectangle(purpRoam, purpRoam, this.getSyncArea().width + purpRoam, this.getSyncArea().height + purpRoam), gui.getColor3());
        gui.r3.addListener(purple);
        gui.g3.addListener(purple);
        gui.b3.addListener(purple);
        groups.add(purple);
        purple.add(new Metaball(30 * ELLIPSE_SCALE));
        purple.add(new Metaball(40 * ELLIPSE_SCALE));
        purple.add(new Metaball(50 * ELLIPSE_SCALE));
        purple.add(new Metaball(50 * ELLIPSE_SCALE));
        purple.add(new Metaball(60 * ELLIPSE_SCALE));
        purple.add(new Metaball(70 * ELLIPSE_SCALE));

        // probably should be in ball constructors
        for (MetaballGroup group : groups){
            for (Metaball ball : group.balls){
                ball.position  = new PVector(random(0, this.getSyncArea().width), random (0, this.getSyncArea().height));
                ball.velocity = new PVector(random(-1,1), random(-1,1));
            }
        }
    }

    @Override
    public void drawELUContent() {

        // move balls
        for (MetaballGroup group : groups){

            PVector center = new PVector(0, 0, 0);

            for (Metaball ball : group.balls){

                boolean runningAway = false;

                if(mode == MOUSE && mousePressed) {
                    ball.repell(new PVector(mouseX, mouseY, 0), gui.getRepelForce());
                    ball.spaceCheck(groups);
                    runningAway = true;

                } else if (mode == GRID && gridData != null) {
                    synchronized(gridData){
                        List<Point> points = this.getObjects(gridData);
                        runningAway = points.size() > gui.getThreshold();
                        for (Point point : points){
                            ball.repell(new PVector(point.x, point.y, 0), gui.getRepelForce());
                        }
                    }
                }

                center.add(ball.position);
                ball.velocity.mult(FRICTION);
                ball.velocity.limit(runningAway ? MAX_RUN_VEL : MAX_VEL);

                if (ball.velocity.mag() < MIN_VEL){
                    ball.velocity.setMag(MIN_VEL);
                }
            }

            center.div(group.balls.size());

            for (Metaball ball : group.balls){
                PVector c = PVector.sub(center, ball.position);
                c.normalize();
                c.mult(gui.getCohesiveness());
                ball.velocity.add(c);
                ball.position.add(ball.velocity);
            }

            group.checkBounds();
        }

        // fill the whole area with purple
        fill(gui.getBGColor().r, gui.getBGColor().g, gui.getBGColor().b, 127);
        rect(0, 0, width, height);

        // render each group's bounding box
        for (MetaballGroup group : groups){
            this.stroke(255,255,255);
            this.noFill();
            this.rect(group.range.x, group.range.y, group.range.width, group.range.height);
        }

        // render groups of balls
        for (MetaballGroup group : groups){
            for (Metaball ball : group.balls){
                this.noStroke();
                this.fill(group.color.r, group.color.g, group.color.b, ELLIPSE_ALPHA);
                this.ellipse(ball.position.x, ball.position.y, ball.width(), ball.height());
            }
        }

        // presence
        if (mode == MOUSE && mousePressed){
            fill(10, 200, 255);
            ellipse(mouseX, mouseY, 120, 120);
        } else if (mode == GRID && gridData !=null) {

            fill(color(0, 0, 50), 8); //fill with a light alpha white
            rect(0, 0, this.getSyncArea().width, this.getSyncArea().height); //fill the whole area

            if (gridData != null){

                synchronized(gridData){

                    gridOnCanvas = new Rectangle(80, 70, 
                            this.getSyncArea().width - 160, 
                            this.getSyncArea().height - 140);
                    float cellWidth = gridOnCanvas.width / (float)gridData.width;
                    float cellHeight = gridOnCanvas.height / (float)gridData.height;

                    if (presenceMode != GRID_DEFAULT){
                        stroke(255);
                    }else{
                        noStroke();
                        fill(255, 255, 255, 127);
                    }

                    for (int x = 0; x < gridData.width; x++){
                        for (int y = 0; y < gridData.height; y++){
                            if (presenceMode != GRID_DEFAULT){
                                if (gridData.getValue(x, y) != (byte)0){
                                    fill(255);
                                }else{
                                    noFill();
                                }
                                this.rect(gridOnCanvas.x + (x * cellWidth), 
                                          gridOnCanvas.y + (y * cellHeight), 
                                          cellWidth, 
                                          cellHeight);
                            } else {
                                if (gridData.getValue(x, y) != (byte)0){
                                    this.ellipse(gridOnCanvas.x + (x * cellWidth), 
                                                 gridOnCanvas.y + (y * cellHeight), 
                                                 cellWidth * DILATE, 
                                                 cellHeight * DILATE);
                                }
                            }
                        }
                    }

                    // render the presences
                }
            }
        }

        loadPixels();
        FastBlur.performBlur(pixels, width, height, floor(18));
        updatePixels();

    }

    public List<Point> getObjects(GridData grid){
        List<Point>objects = new ArrayList<Point>();
        for (int i = 0; i < grid.data.length; i++){
            if (grid.data[i] != (byte)0){
                int y = i / grid.width;
                int x = i - (y * grid.width);
                objects.add(new Point(x,y));
            }
        }
        return objects;
    }

    @Override
    public void keyPressed() {
        super.keyPressed();
        System.out.println("key:  " + key);
        System.out.println("code: " + keyCode);
    }

    @Override
    public void handle(GridData srcData) {
        if (gridData == null){
            gridData = srcData;
        } else {
            synchronized(gridData){

                // copy the original source, so we don't accidentally change
                // the source for other clients using this data.
                StringBuilder sb = new StringBuilder();
                srcData.buildString(sb);
                srcData = new GridData(sb.toString());

                srcData = subset(srcData, grid);

                srcData = counterClockwise(srcData);

                srcData = flipHorizontal(srcData);

                gridData = srcData;
            }
        }
    }

    /**
     * returns new GridData where data is a subset of the original based on
     * a Rectangular boundary.
     * 
     * @param in
     * @param boundary
     * @return
     */
    public static GridData subset(GridData in, Rectangle boundary){

        byte[] target = new byte[boundary.width * boundary.height];
        for (int y = 0; y < boundary.height; y++) {
            System.arraycopy(in.data, ((y + boundary.y) * in.width) + (boundary.x), target, y * boundary.width, boundary.width);
        }
        in.data  = target;
        in.height = boundary.height;
        in.width  = boundary.width;
        return in;
    }

    public static GridData counterClockwise(GridData in){

        // switch height for width and vice versa
        byte[] rotated = new byte[in.data.length];
        int i = 0;

        for (int x = in.width - 1; x >= 0; x--){
            for (int y = 0; y < in.height; y++){
                rotated[i++] = in.data[y * in.width + x];
            }
        }
        int w = in.width;
        in.width = in.height;
        in.height = w;
        in.data = rotated;

        return in;
    }

    public static GridData flipHorizontal(GridData in){
        int center = in.width / 2;
        byte buffer;

        for (int y = 0; y < in.height; y++){
            for (int x = 0; x < center; x++){
                int leftIndex = y * in.width + x;
                int rightIndex = (y + 1) * in.width - x - 1;
                buffer = in.data[leftIndex];
                in.data[leftIndex] = in.data[rightIndex];
                in.data[rightIndex] = buffer;
            }
        }
        return in;
    }
/**
    @Override
    public void handle(List<BaseTrack> incomingTracks){
        if (trackData == null){
            trackData = incomingTracks;
        } else {
            synchronized(trackData){
                trackData = incomingTracks;
                List<BaseTrack> transformedTrack;
    
                float cameraWidth = 544;
                float cameraHeight = 480;
                float gridWidth = 80;
                float gridHeight = 40;
                int gridXMax = 70;
                int gridXMin = 2;
                int gridYMax = 25;
                int gridYMin = 2;
                int canvasInsetX = 80;
                int canvasInsetY =70;
                float subsetWidth = (cameraWidth / gridWidth) * (gridXMax - gridXMin); // grid insets (left & right)
                float subsetHeight = (cameraHeight / gridHeight) * (gridYMax - gridYMin);  // grid insets (top & bottom)
                float subsetLeftInset = (cameraWidth / gridWidth) * 2;
                float subsetTopInset = (cameraWidth / gridHeight) * 2;
                float finalWidth = this.getSyncArea().width - (2 * canvasInsetX); // 80 = canvas x inset
                float finalHeight = this.getSyncArea().height - (2 * canvasInsetY); // 70 = canvas y inset
                float finalWidthFactor =  finalWidth / subsetWidth;
                float finalHeightFactor =  finalHeight / subsetHeight;

                // take subset
                transformedTrack = subset(incomingTracks, new Rectangle((int)subsetLeftInset, (int)subsetTopInset , (int)subsetWidth, (int)subsetHeight));

                // rotate around center of subset
                transformedTrack = rotate(incomingTracks, -90, new Point((int)(subsetWidth / 2), (int)(subsetHeight / 2)));

                // scale to size of detector system
                transformedTrack = scale(incomingTracks, new PVector(finalWidthFactor, finalHeightFactor, 0));

                // mirror
                transformedTrack = flipHorizontal(incomingTracks, finalWidth);

                // inset to detector system top,left
                transformedTrack = slide(incomingTracks, new PVector(80, 70, 0));


                trackData = transformedTrack;
            }
        }
    }

    // pull out all tracks from a subset of the space they were originally contained in
    // using the same coordinate system.  The resulting subset will be boundary.width x boundary.height in dimensions, 
    // with all tracks shifted to a coordinate space where the top left is 0,0 (so that future rotations don't go amuck)
    public static List<BaseTrack> subset(List<BaseTrack> in, Rectangle boundary){
        ArrayList<BaseTrack>out = new ArrayList<BaseTrack>();
        for (BaseTrack track : in){
            if (boundary.contains(track.x, track.y)){
                track.x -= boundary.x;
                track.y -= boundary.y;
                out.add(track);
            }
        }
        return out;
    }

    // rotate around an arbitraty point.  negative = clockwise
    public static List<BaseTrack> rotate(List<BaseTrack> in, float degrees, Point center){
        for (BaseTrack track : in){
            float[] pt = {track.x, track.y};
            AffineTransform.getRotateInstance(Math.toRadians(degrees), center.x, center.y).transform(pt, 0, pt, 0, 1);
            track.x = pt[0];
            track.y = pt[1];
        }
        return in;
    }

    // scale the coordinates of all tracks
    public static List<BaseTrack> scale(List<BaseTrack> in, PVector scaleValues){
        for (BaseTrack track : in){
            track.x *= scaleValues.x;
            track.y *= scaleValues.y;
        }
        return in;
    }

    // slide the coordinates of all tracks
    public static List<BaseTrack> slide(List<BaseTrack> in, PVector slideValues){
        for (BaseTrack track : in){
            track.x += slideValues.x;
            track.y += slideValues.y;
        }
        return in;
    }

    // slide the coordinates of all tracks
    public static List<BaseTrack> flipHorizontal(List<BaseTrack> in, float rightEdge){
        for (BaseTrack track : in){
            track.x = rightEdge - track.x;
        }
        return in;
    }
*/
    class MetaballGroup implements ControlListener{
        
        Rectangle range;
        PVector position;
        PVector velocity;
        Color color;
        List <Metaball>balls;

        public MetaballGroup(Rectangle range, Color color){
            this.range = range;
            this.color = color;
            balls = new ArrayList<Metaball>();
        }

        public void add(Metaball ball){
            ball.group = this;
            balls.add(ball);
        }
        
        public void checkBounds(){
            for (Metaball ball : balls){
                ball.checkBounds(range);
            }
        }

        // hacky code to get events from controlP5
        @Override 
        public void controlEvent(ControlEvent evt) {
            if (evt.getController().getName().toLowerCase().startsWith("red")){
                this.color.r = evt.getValue();
            }else if (evt.getController().getName().toLowerCase().startsWith("green")){
                this.color.g = evt.getValue();
            }else if (evt.getController().getName().toLowerCase().startsWith("blue")){
                this.color.b = evt.getValue();
            }
        }
    }

    class Metaball {

        float radius;

        PVector position;
        PVector velocity;
        float xsquish = 0;
        float ysquish = 0;
        MetaballGroup group;

        public Metaball(float radius){
            this.radius = radius;
        }

        public void checkBounds(Rectangle range){
            // bounce
            if (left() < range.x){
                setLeft(range.x);
                velocity.x = -velocity.x;
            } else if (right() > range.width + range.x){
                setRight(range.width + range.x);
                velocity.x = -velocity.x;
            }
            if (top() < range.y){
                setTop(range.y);
                velocity.y = -velocity.y;
            } else if (bottom() > range.height + range.y){
                setBottom(range.height + range.y);
                velocity.y = -velocity.y;
            }
        }

        public void repell(PVector point, float force){
            
            float mDistance = PVector.dist(position, point);
            PVector repel = PVector.sub(position, point);
            repel.normalize();
            repel.mult(force / (mDistance * mDistance));
            velocity.add(repel);
            
        }

        public void spaceCheck(List<MetaballGroup> groups){
            for (MetaballGroup group : groups) {
                if (group != this.group) {
                    for (Metaball ball : group.balls){
                        if (ball.overlaps(this) || ball.group != this.group){
                            this.repell(ball.position, Metaballs2.BALL_REPELL_FORCE);
                        }
                    }
                }
            }
        }
        
        public boolean overlaps(Metaball ball){
            float distance = ball.position.dist(this.position);
            return distance < ball.radius || distance < this.radius;
        }
        
        public float width(){
            return radius - xsquish;
        }
        public float height(){
            return radius - ysquish;
        }

        public float left(){
            return position.x - width() / 2;
        }
        public float right(){
            return position.x + width() / 2;
        }
        public float top(){
            return position.y - height() / 2;
        }
        public float bottom(){
            return position.y + height() / 2;
        }

        public void setLeft(float left){
            position.x = width() / 2 + left; 
        }
        public void setRight(float right){
            position.x = right - width() / 2;
        }
        public void setTop(float top){
            position.y = top + height() / 2;
        }
        public void setBottom(float bottom){
            position.y = bottom - height() / 2;
        }
    }


}