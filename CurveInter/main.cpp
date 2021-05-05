//
//  main.cpp
//  CurveInter
//
//  Created by Hyun Joon Shin on 2021/03/24.
//

#include <iostream>
#include <JGL/JGL.hpp>
#include <JGL/JGL_Window.hpp>
#include <JGL/JGL_Widget.hpp>
#include <JGL/JGL_Options.hpp>
#include <JGL/JGL_Toolbar.hpp>
#include <JGL/JGL_Aligner.hpp>

#include <Eigen/Dense>

enum {
	LAGLANGIAN,
	BEZIER,
	CATMULL_ROM,
	CLOSED,
	NATURAL,
	CATMULL,
	BSPLINE,
	LINEAR,
};

enum {
	DRAW_LINES,
	DRAW_DOTS,
};

int curveType;
int drawType;
std::vector<glm::vec2> srcPts;
bool pointsDirty = true;


glm::vec2 evaluateCurve( size_t k, float t ) {
	
	glm::vec2 ret(0,0);

	switch( curveType ) {
		case LAGLANGIAN: {
			float T = k+t;
			for( auto i=0; i<srcPts.size(); i++ ) {
				float L = 1;
				for( auto j=0; j<srcPts.size(); j++ ) {
					if( j!=i ) {
						L/=(i-j);
						L*=T-j;
					}
				}
				ret+=L*srcPts[i];
			}
		} break;
		case BEZIER: {
			if( k>2 )
				ret = srcPts[k];
			else {
				float T = (k+t)/3;
				ret = (1-T)*(1-T)*(1-T)*srcPts[0]
				+3*(1-T)*(1-T)*T*srcPts[1]
				+3*(1-T)*T*T*srcPts[2]
				+T*T*T*srcPts[3];
			}
			
		} break;
		case CATMULL_ROM: {
			glm::vec2 p0, p1, p2, p3;
			glm::vec2 v0(0), v1(0);
			
			if (k > 0) {
				glm::mat3 A((k - 1) * (k - 1), k * k, (k + 1) * (k + 1), k - 1, k, k + 1, 1, 1, 1);
				glm::vec3 bx = glm::vec3(srcPts[k - 1].x, srcPts[k].x, srcPts[k + 1].x);
				glm::vec3 by = glm::vec3(srcPts[k - 1].y, srcPts[k].y, srcPts[k + 1].y);
				glm::vec3 xx = glm::inverse(A) * bx;
				glm::vec3 xy = glm::inverse(A) * by;
				v0 = glm::vec2(2 * k * xx.x + xx.y, 2 * k * xy.x + xy.y);
			}

			if (k < srcPts.size() - 2) {
				glm::mat3 A(k * k, (k + 1) * (k + 1), (k + 2) * (k + 2), k, k + 1, k + 2, 1, 1, 1);
				glm::vec3 bx = glm::vec3(srcPts[k].x, srcPts[k + 1].x, srcPts[k + 2].x);
				glm::vec3 by = glm::vec3(srcPts[k].y, srcPts[k + 1].y, srcPts[k + 2].y);
				glm::vec3 xx = glm::inverse(A) * bx;
				glm::vec3 xy = glm::inverse(A) * by;
				v1 = glm::vec2(2 * (k + 1) * xx.x + xx.y, 2 * (k + 1) * xy.x + xy.y);
			}

			if (k == 0) {
				glm::mat3 A(0, 1, 4, 0, 1, 2, 1, 1, 1);
				glm::vec3 bx = glm::vec3(srcPts[0].x, srcPts[1].x, srcPts[2].x);
				glm::vec3 by = glm::vec3(srcPts[0].y, srcPts[1].y, srcPts[2].y);
				glm::vec3 xx = glm::inverse(A) * bx;
				glm::vec3 xy = glm::inverse(A) * by;
				v0 = glm::vec2(xx.y, xy.y);
			}
			
			if (k == srcPts.size() - 2) {
				glm::mat3 A((k - 1) * (k - 1), k * k, (k + 1) * (k + 1), k - 1, k, k + 1, 1, 1, 1);
				glm::vec3 bx = glm::vec3(srcPts[k - 1].x, srcPts[k].x, srcPts[k + 1].x);
				glm::vec3 by = glm::vec3(srcPts[k - 1].y, srcPts[k].y, srcPts[k + 1].y);
				glm::vec3 xx = glm::inverse(A) * bx;
				glm::vec3 xy = glm::inverse(A) * by;
				v1 = glm::vec2(2 * (k + 1) * xx.x + xx.y, 2 * (k + 1) * xy.x + xy.y);
			}
			

			p0 = srcPts[k];
			p3 = srcPts[k + 1];
			p1 = p0 + v0 / 3.f;
			p2 = p3 - v1 / 3.f;
			ret = (1 - t) * (1 - t) * (1 - t) * p0
				+ 3 * (1 - t) * (1 - t) * t * p1
				+ 3 * (1 - t) * t * t * p2
				+ t * t * t * p3;

		} break;
		
		case CLOSED: {

			static size_t COMPUTED_N = 0;
			static Eigen::MatrixXf Ainv;

			size_t N = srcPts.size() - 1;

			if (N != COMPUTED_N) {
				Eigen::FullPivLU<Eigen::MatrixXf> solver;
				Eigen::MatrixXf A(N + 1, N + 1);
				A.setZero();
				A(0, 0) = 4;
				A(0, 1) = 1;
				A(0, N) = 1;

				for (auto i = 1; i < N; i++) {
					A(i, i - 1) = 1;
					A(i, i) = 4;
					A(i, i + 1) = 1;
				}

				A(N, 0) = 1;
				A(N, N - 1) = 1;
				A(N, N) = 4;

				solver = A.fullPivLu();
				Ainv = solver.inverse();
				COMPUTED_N = N;
			}

			Eigen::VectorXf bx(N + 1), by(N + 1);

			bx[0] = 3 * (srcPts[1].x - srcPts[N].x);
			by[0] = 3 * (srcPts[1].y - srcPts[N].y);

			for (auto i = 1; i < N; i++) {
				bx[i] = 3 * (srcPts[i + 1].x - srcPts[i - 1].x);
				by[i] = 3 * (srcPts[i + 1].y - srcPts[i - 1].y);
			}

			bx[N] = 3 * (srcPts[0].x - srcPts[N - 1].x);
			by[N] = 3 * (srcPts[0].y - srcPts[N - 1].y);

			auto Dx = Ainv * bx;
			auto Dy = Ainv * by;
			
			glm::vec2 a, b, c, d;
			
			
			if (k > srcPts.size() - 2) {
				a = srcPts[k];
				b = glm::vec2(Dx[k], Dy[k]);
				c = 3.f * (srcPts[0] - srcPts[k]) - 2.f * glm::vec2(Dx[k], Dy[k]) - glm::vec2(Dx[0], Dy[0]);
				d = 2.f * (srcPts[k] - srcPts[0]) + glm::vec2(Dx[k], Dy[k]) + glm::vec2(Dx[0], Dy[0]);
			}
			else {
				a = srcPts[k];
				b = glm::vec2(Dx[k], Dy[k]);
				c = 3.f * (srcPts[k + 1] - srcPts[k]) - 2.f * glm::vec2(Dx[k], Dy[k]) - glm::vec2(Dx[k + 1], Dy[k + 1]);
				d = 2.f * (srcPts[k] - srcPts[k + 1]) + glm::vec2(Dx[k], Dy[k]) + glm::vec2(Dx[k + 1], Dy[k + 1]);
			}
			
			ret = a + b * t + c * t * t + d * t * t * t;
			
		} break;

		case NATURAL: {

			static size_t COMPUTED_N = 0;
			static Eigen::MatrixXf Ainv;
			
			size_t N = srcPts.size() - 1;
			
			if (N != COMPUTED_N) {

				Eigen::FullPivLU<Eigen::MatrixXf> solver;
				Eigen::MatrixXf A(N + 1, N + 1);

				A.setZero();
				A(0, 0) = 2;
				A(0, 1) = 1;

				for (auto i = 1; i < N; i++) {
					A(i, i-1) = 1;
					A(i, i) = 4;
					A(i, i+1) = 1;
				}

				A(N, N - 1) = 1;
				A(N, N) = 2;

				solver = A.fullPivLu();
				Ainv = solver.inverse();
				COMPUTED_N = N;
			}

			Eigen::VectorXf bx(N + 1);
			Eigen::VectorXf by(N + 1);

			bx[0] = 3 * (srcPts[1].x - srcPts[0].x);
			by[0] = 3 * (srcPts[1].y - srcPts[0].y);

			for (auto i = 1; i < N; i++) {
				bx[i] = 3 * (srcPts[i+1].x - srcPts[i-1].x);
				by[i] = 3 * (srcPts[i+1].y - srcPts[i-1].y);
			}

			bx[N] = 3 * (srcPts[N].x - srcPts[N - 1].x);
			by[N] = 3 * (srcPts[N].y - srcPts[N - 1].y);

			auto Dx = Ainv*bx;
			auto Dy = Ainv*by;
/*
			Eigen::VectorXf Dx = Eigen::VectorXf::Zero(N + 1);
			Eigen::VectorXf Dy = Eigen::VectorXf::Zero(N + 1);
			for (auto iter = 0; iter < 10; iter++) {
				Dx[0] = (3 * (srcPts[1].x - srcPts[0].x) - Dx[1]) / 2;
				Dy[0] = (3 * (srcPts[1].y - srcPts[0].y) - Dy[1]) / 2;
				for (auto i = 1; i < N; i++) {
					auto bx = 3 * (srcPts[i + 1].x - srcPts[i - 1].x);
					Dx[i] = (bx - Dx[i - 1] - Dx[i + 1]) / 4;
					auto by = 3 * (srcPts[i + 1].y - srcPts[i - 1].y);
					Dy[i] = (by - Dy[i - 1] - Dy[i + 1]) / 4;
				}
				Dx[N] = (3 * (srcPts[N].x - srcPts[N - 1].x) - Dx[N - 1]) / 2;
				Dy[N] = (3 * (srcPts[N].y - srcPts[N - 1].y) - Dy[N - 1]) / 2;
			}
*/
			glm::vec2 a = srcPts[k];
			glm::vec2 b = glm::vec2(Dx[k], Dy[k]);
			glm::vec2 c = 3.f * (srcPts[k + 1] - srcPts[k]) - 2.f * glm::vec2(Dx[k], Dy[k]) - glm::vec2(Dx[k + 1], Dy[k + 1]);
			glm::vec2 d = 2.f * (srcPts[k] - srcPts[k + 1]) + glm::vec2(Dx[k], Dy[k]) + glm::vec2(Dx[k + 1], Dy[k + 1]);
			return a + b * t + c * t * t + d * t * t * t;

		} break;

		case CATMULL: {
			glm::vec2 p0, p1, p2, p3;
			glm::vec2 v0(0), v1(0);

			if (k > 0) {
				v0 = (srcPts[k + 1] - srcPts[k - 1]) / 2.f;
			}

			if (k < srcPts.size() - 2) {
				v1 = (srcPts[k + 2] - srcPts[k]) / 2.f;
			}

			p0 = srcPts[k];
			p3 = srcPts[k + 1];
			p1 = p0 + v0 / 3.f;
			p2 = p3 - v1 / 3.f;
			ret = (1 - t) * (1 - t) * (1 - t) * p0
				+ 3 * (1 - t) * (1 - t) * t * p1
				+ 3 * (1 - t) * t * t * p2
				+ t * t * t * p3;

		} break;

		case BSPLINE: {
			if (k<2 || k>srcPts.size() - 3) {
				ret = srcPts[k];
			}
			else {
				float B0 = 1.f / 6.f * (1 - t) * (1 - t) * (1 - t);
				float B1 = 1.f / 6.f * (3 * t * t * t - 6 * t * t + 4);
				float B2 = 1.f / 6.f * (-3 * t * t * t + 3 * t * t + 3 * t + 1);
				float B3 = 1.f / 6.f * t * t * t;
				ret = B0 * srcPts[k - 1] + B1 * srcPts[k] + B2 * srcPts[k + 1] + B3 * srcPts[k + 2];
			}
		} break;

		case LINEAR :
		default: {
			ret = (srcPts[k+1]-srcPts[k])*t + srcPts[k];
		}
	}
	pointsDirty = false;
	return ret;
}



struct CurveWidget : JGL::Widget {
	CurveWidget(float x, float y, float w, float h, const std::string& title = "" )
	: JGL::Widget(x,y,w,h,title){}
	virtual void		drawBox(NVGcontext* vg, const glm::rect& r) override {
		nvgBeginPath( vg );
		nvgRect( vg, r.x, r.y, r.w, r.h );
		nvgFillColor( vg, nvgRGBAf(0,0,0,1));
		nvgFill( vg );
	}

	virtual void drawContents(NVGcontext* vg, const glm::rect& r, int align ) override {
		
		if( drawType == DRAW_LINES ) {
			nvgBeginPath( vg );
			for( auto i=0; i<srcPts.size()-(curveType==CLOSED?0:1); i++ ) {
				for( float t=0; t<1.0; t+=0.01f ) {
					glm::vec2 p = evaluateCurve( i, t );
						if( i==0 && t<0.0001 )
							nvgMoveTo( vg, p.x, p.y );
						else
							nvgLineTo( vg, p.x, p.y );
				}
			}
			nvgStrokeColor(vg, nvgRGBAf(0,.3f,1,1));
			nvgStrokeWidth(vg, 2);
			nvgStroke( vg );
		}
		else {
			nvgBeginPath( vg );
			for( auto i=0; i<srcPts.size()-1; i++ ) {
				for( float t=0; t<1.0; t+=0.01f ) {
					glm::vec2 p = evaluateCurve( i, t );
					nvgCircle(vg, p.x, p.y, 1);
				}
			}
			nvgFillColor(vg, nvgRGBAf(0,1,.3f,1));
			nvgFill( vg );
		}

		
		nvgBeginPath( vg );
		for( auto i=0; i<srcPts.size(); i++ )
			if( i!= underPt )
				nvgCircle( vg, srcPts[i].x, srcPts[i].y, 5 );
		nvgFillColor( vg, nvgRGBAf(1,1,0,.8f));
		nvgFill( vg );

		if( underPt>=0 ) {
			nvgBeginPath( vg );
			nvgCircle( vg, srcPts[underPt].x, srcPts[underPt].y, 5 );
			nvgFillColor( vg, nvgRGBAf(1,.8f,0,.8f));
			nvgFill( vg );
		}

	}
	virtual bool handle( int event ) override {
		glm::vec2 pt = JGL::_JGL::eventPt();
		switch( event ) {
			case JGL::EVENT_MOVE : {
				int oldPt = underPt;
				underPt = -1;
				for( auto i=0; i<srcPts.size(); i++ ) {
					if( length(pt-srcPts[i])<6 )
						underPt = i;
				}
				if( underPt!= oldPt )
					redraw();

			}break;
			case JGL::EVENT_PUSH : {
				if( underPt>=0 )
					ptOffset = srcPts[underPt]-pt;
			}break;
			case JGL::EVENT_DRAG : {
				if( underPt>=0 ) {
					srcPts[underPt] = pt+ptOffset;
					pointsDirty = true;
					redraw();
				}
			}break;
		}
		return true;
	}
	int underPt = -1;
	glm::vec2 ptOffset = glm::vec2(0);
};


using namespace JGL;
CurveWidget* curveWidget;

void curveTypeCallback(Widget* w, void* ud ) {
	curveType = ((Options*)w)->value();
	curveWidget->redraw();
}

void drawTypeCallback(Widget* w, void* ud ) {
	drawType = ((Options*)w)->value();
	curveWidget->redraw();
}

int main(int argc, const char * argv[]) {
	// insert code here...
	std::cout << "Hello, World!\n";
	
	/*
	Eigen::Matrix<float, 3, 3> A; A << 1, 3, 2, 1, 2, 3, 5, 3, 1;
	Eigen::Vector3f b; b << 1, 2, 3;
	auto x = A.fullPivLu().solve(b);
	cout << x(0) << " " << x(1) << " " << x(2) << endl;
	auto b2 = A * x;
	cout << b2(0) << " " << b2(1) << " " << b2(2) << endl;
	*/
	Window* window = new Window(640,480,"Curves");
	window->alignment(ALIGN_ALL);
	
	Aligner* aligner = new Aligner(0,0,window->w(), window->h());
	aligner->type(Aligner::VERTICAL);
	aligner->alignment(ALIGN_ALL);

	Toolbar* toolbar = new Toolbar(0,0,window->w(), _size_toolbar_height() );
		Options* curveType = new Options(0,0,200,_size_button_height() );
		curveType->add("Lagrangian");
		curveType->add("Bezier");
		curveType->add("Catmull_Rom");
		curveType->add("Closed");
		curveType->add("Natural");
		curveType->add("Catmull");
		curveType->add("Bspline");
		curveType->add("Linear");
		curveType->value(::curveType);
		curveType->callback( curveTypeCallback );
		Options* drawType = new Options(0,0,200,_size_button_height() );
		drawType->add("Lines");
		drawType->add("Dots");
		drawType->value(::drawType);
		drawType->callback( drawTypeCallback );
	toolbar->end();
	
	curveWidget = new CurveWidget(0,0,window->w(), window->h()-toolbar->h());
	aligner->resizable( curveWidget );
	aligner->end();

	for( auto i=0; i<9; i++ ) {
		srcPts.push_back(glm::vec2((i*0.1+0.1)*curveWidget->w(),curveWidget->h()/2));
	}

	window->show();
	_JGL::run();
	
	return 0;
}
