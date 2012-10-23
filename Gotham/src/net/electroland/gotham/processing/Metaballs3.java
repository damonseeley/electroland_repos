package net.electroland.gotham.processing;

import java.awt.Point;
import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.List;

import net.electroland.elvis.net.GridData;
import net.electroland.gotham.processing.assets.Color;
import net.electroland.gotham.processing.assets.FastBlur;
import net.electroland.gotham.processing.assets.MetaballsGUI;
import processing.core.PVector;
import controlP5.ControlEvent;
import controlP5.ControlListener;

@SuppressWarnings("serial")
public class Metaballs3 extends GothamPApplet {

    public static final int NONE                    = 0;
    public static final int MOUSE                   = 1;
    public static final int GRID                    = 2;

    private int mode = GRID; // set the reaction mode

    public static final float FRICTION            = .999f;  // higher = less .999
    public static final float MAX_VEL             = 1.0f;  // base velocity with no interaction or tracks higher = faster was 0.75
    // push vars
    public static final float MAX_RUN_VEL         = 1000.0f;  // max velocity when mouse is down or presence is felt.  //30
    // ball group props
    public static final float BALL_REPELL_FORCE   = 20;      // group to group repell force (higher = more)
    // ball scale
    public static final float ELLIPSE_SCALE       = 5.0f;   // percent 2.0-2.4 then 3.5 for most testting
    public static final int ELLIPSE_ALPHA         = 150;   // value/255
    // presence scale
    public static final float DILATE              = 6.0f;

    // don't touch:
    public static final float MIN_VEL             = .75f;   // higher = faster

    private List <MetaballGroup>groups;
    private GridData gridData;
    private MetaballsGUI prefs;


    @Override
    public void setup(){

        prefs = new MetaballsGUI(this);

        // groups of balls
        groups = new ArrayList<MetaballGroup>();

        int redOrgRoam = 40; //was 100
        int purpRoam = 0; //was 0

        MetaballGroup red = new MetaballGroup(new Rectangle(-redOrgRoam, -redOrgRoam, this.getSyncArea().width + redOrgRoam, this.getSyncArea().height +redOrgRoam), prefs.getColor1());
        prefs.r1.addListener(red);
        prefs.g1.addListener(red);
        prefs.b1.addListener(red);
        groups.add(red);
        red.add(new Metaball(75 * ELLIPSE_SCALE));
        red.add(new Metaball(80 * ELLIPSE_SCALE));
        red.add(new Metaball(100 * ELLIPSE_SCALE));
        red.add(new Metaball(75 * ELLIPSE_SCALE));
        red.add(new Metaball(80 * ELLIPSE_SCALE));
        red.add(new Metaball(100 * ELLIPSE_SCALE));

        MetaballGroup orange = new MetaballGroup(new Rectangle(-redOrgRoam, -redOrgRoam, this.getSyncArea().width + redOrgRoam, this.getSyncArea().height +redOrgRoam), prefs.getColor2());
        prefs.r2.addListener(orange);
        prefs.g2.addListener(orange);
        prefs.b2.addListener(orange);
        groups.add(orange);
        orange.add(new Metaball(70 * ELLIPSE_SCALE));
        orange.add(new Metaball(80 * ELLIPSE_SCALE));
        orange.add(new Metaball(90 * ELLIPSE_SCALE));
        orange.add(new Metaball(70 * ELLIPSE_SCALE));
        orange.add(new Metaball(80 * ELLIPSE_SCALE));
        //orange.add(new Metaball(90 * ELLIPSE_SCALE));

        MetaballGroup purple = new MetaballGroup(new Rectangle(purpRoam, purpRoam, this.getSyncArea().width + purpRoam, this.getSyncArea().height + purpRoam), prefs.getColor3());
        prefs.r3.addListener(purple);
        prefs.g3.addListener(purple);
        prefs.b3.addListener(purple);
        groups.add(purple);
        purple.add(new Metaball(30 * ELLIPSE_SCALE));
        purple.add(new Metaball(40 * ELLIPSE_SCALE));
        purple.add(new Metaball(50 * ELLIPSE_SCALE));
        purple.add(new Metaball(50 * ELLIPSE_SCALE));
        purple.add(new Metaball(60 * ELLIPSE_SCALE));
        //purple.add(new Metaball(60 * ELLIPSE_SCALE));

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
                    ball.repell(new PVector(mouseX, mouseY, 0), prefs.getRepelForce());
                    ball.spaceCheck(groups);
                    runningAway = true;

                } else if (mode == GRID && gridData != null) {
                    synchronized(gridData){
                        List<Point> points = this.getObjects(gridData);
                        runningAway = points.size() > prefs.getThreshold();
                        for (Point point : points){

                            float cellWidth = prefs.getGridCanvas().width / (float)gridData.width;
                            float cellHeight = prefs.getGridCanvas().height / (float)gridData.height;

                            PVector translated = new PVector((cellWidth * point.x) + prefs.getGridCanvas().x, 
                                                         (cellHeight * point.y)  + prefs.getGridCanvas().y);

                            ball.repell(translated, prefs.getRepelForce());
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
                c.mult(prefs.getCohesiveness());
                ball.velocity.add(c);
                ball.position.add(ball.velocity);
            }

            group.checkBounds();
        }

        // fill the whole area with purple
        fill(prefs.getBGColor().r, prefs.getBGColor().g, prefs.getBGColor().b, 127);
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

                    float cellWidth = prefs.getGridCanvas().width / (float)gridData.width;
                    float cellHeight = prefs.getGridCanvas().height / (float)gridData.height;

                    if (prefs.showGrid()){
                        stroke(255);
                    }else{
                        noStroke();
                        fill(255, 255, 255, 127);
                    }

                    for (int x = 0; x < gridData.width; x++){
                        for (int y = 0; y < gridData.height; y++){
                            if (prefs.showGrid()){
                                if (gridData.getValue(x, y) != (byte)0){
                                    fill(255);
                                }else{
                                    noFill();
                                }
                                this.rect(prefs.getGridCanvas().x + (x * cellWidth), 
                                        prefs.getGridCanvas().y + (y * cellHeight), 
                                        cellWidth, 
                                        cellHeight);
                            } else {
                                if (gridData.getValue(x, y) != (byte)0){
                                    this.ellipse(prefs.getGridCanvas().x + (x * cellWidth), 
                                            prefs.getGridCanvas().y + (y * cellHeight), 
                                                 //cellWidth * DILATE, 
                                                 //cellHeight * DILATE);
                                    			80,80);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (!prefs.showGrid()){
            loadPixels();
            FastBlur.performBlur(pixels, width, height, floor(20));
            updatePixels();
        }
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

                srcData = subset(srcData, prefs.getGrid());

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