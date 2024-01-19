
//librería glfw

#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#include<stdio.h>


static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}


#define     WIDTH    800
#define		HEIGHT	 600
#define EPSILON			0.005f


GLuint  bufferObj;
cudaGraphicsResource *resource;

int cambiofx = 1;
int cambiofy = 1;


#define NSPHERES 3
#define NPLANES 1
#define NLIGHTS 2


//spheres Code 25
#define SPHERE 25
__constant__ float sphere[NSPHERES * 4];  //NSPHERES spheres will be stored as cx,cy,cz,r.
__constant__ float sphereColor[NSPHERES * 4]; //NSPHERES colors will be stored as r,g,b,refraction.
__constant__ float sphereLightProperties[NSPHERES * 2]; //NSPHERES properties will be stored as diff, reflection.

//planes Code 26
#define PLANE 26
__constant__ float plane[NPLANES * 4]; //NPLANES plane will be stored as Nx Ny Nz D.
__constant__ float planeColor[NPLANES * 3]; //NPLANES color will be stored as r,g,b.
__constant__ float planeLightProperties[NPLANES * 2]; //NPLANES properties will be stored as diff, reflection.

//lights Code 27
#define LIGHT 27
__constant__ float light[NLIGHTS * 4]; //NLIGHTS light will be stored as cx, cy, cz,r.
__constant__ float lightColor[NLIGHTS * 3]; //NLIGHTS light color will be stored as r,g,b.


//__constant__ float transformation[16];

__device__ void intersections(float originX, float originY, float originZ,
	float dX, float dY, float dZ,
	char* primitive, int* position, float* distance){
	*distance = 1000000.0f; //inf
	*primitive = -1;
	*position = -1;

	float vX, vY, vZ;
	float discriminant;
	float t;

	//check spheres
	for (int i = 0; i<NSPHERES; i++){
		vX = originX - sphere[i * 4];
		vY = originY - sphere[i * 4 + 1];
		vZ = originZ - sphere[i * 4 + 2];

		discriminant = (vX*dX + vY*dY + vZ*dZ)*(vX*dX + vY*dY + vZ*dZ) - (vX*vX + vY*vY + vZ*vZ) + (sphere[i * 4 + 3] * sphere[i * 4 + 3]); //dot(v,d)^2 - (dot(v,v) - r^2)

		if (discriminant > 0){
			t = sqrtf(discriminant);
			t = fminf(-1.0f*(vX*dX + vY*dY + vZ*dZ) + t, -1.0f*(vX*dX + vY*dY + vZ*dZ) - t);
			if (t > 0){
				*distance = fminf(t, *distance);
				if (*distance == t){ //found a closer primitive
					*primitive = SPHERE;
					*position = i;
				}
			}
		}
	}


	
	for (int i = 0; i<NPLANES; i++){
		t = (plane[i * 4] * dX + plane[i * 4 + 1] * dY + plane[i * 4 + 2] * dZ);
		if (t != 0){
			t=-(plane[i * 4] * originX + plane[i * 4 + 1] * originY + plane[i * 4 + 2] * originZ + plane[i * 4 + 3])
				/t ;
			if (t > 0){
				*distance = fminf(t, *distance);
				if (t == *distance){ //found a closer primitive
					*primitive = PLANE;
					*position = i;
				}
			}
		}
	}


	//check lights
	for (int i = 0; i<NLIGHTS; i++){
		vX = originX - light[i * 4];
		vY = originY - light[i * 4 + 1];
		vZ = originZ - light[i * 4 + 2];

		discriminant = (vX*dX + vY*dY + vZ*dZ)*(vX*dX + vY*dY + vZ*dZ) - (vX*vX + vY*vY + vZ*vZ) + (light[i * 4 + 3] * light[i * 4 + 3]); //dot(v,d)^2 - (dot(v,v) - r^2))

		if (discriminant >= 0){
			t = fminf(-1.0f*(vX*dX + vY*dY + vZ*dZ) + sqrtf(discriminant), -1.0f*(vX*dX + vY*dY + vZ*dZ) - sqrtf(discriminant));
			if (t >=0){
				*distance = fminf(t, *distance);
				if (*distance == t){ //found a closer primitive
					*primitive = LIGHT;
					*position = i;
				}
			}
		}
	}

}


__device__ void newRay2(float originX, float originY, float originZ, float vX, float vY, float vZ, float &r, float &g, float &b){
	float intersectionPointX, intersectionPointY, intersectionPointZ;
	float nX, nY, nZ, lX, lY, lZ;
	float rX, rY, rZ;
	float distance;
	float shade = 0.0f;
	float aux;
	char primitiveType;
	int primitivePosition;

	float distanceR;
	char primitiveTypeR;
	int primitivePositionR;

	//Normalize V
	aux = 1 / sqrtf(vX*vX + vY*vY + vZ*vZ);
	vX *= aux;
	vY *= aux;
	vZ *= aux;

	intersections(originX, originY, originZ, vX, vY, vZ, &primitiveType, &primitivePosition, &distance);

	r = 0.0f;
	g = 0.0f;
	b = 0.0f;
	if (primitiveType == LIGHT){ //if it hit a light, assign the color and stop this ray
		r += lightColor[3 * primitivePosition];
		g += lightColor[3 * primitivePosition + 1];
		b += lightColor[3 * primitivePosition + 2];
		//	isActive=false;
	}
	else if (primitiveType != -1 && distance>0){

		intersectionPointX = originX + distance*vX;
		intersectionPointY = originY + distance*vY;
		intersectionPointZ = originZ + distance*vZ;

		if (primitiveType == SPHERE){
			nX = intersectionPointX - sphere[4 * primitivePosition];
			nY = intersectionPointY - sphere[4 * primitivePosition + 1];
			nZ = intersectionPointZ - sphere[4 * primitivePosition + 2];
			//Normalize n
			aux = 1.0f / sqrtf(nX*nX + nY*nY + nZ*nZ);
			nX *= aux;
			nY *= aux;
			nZ *= aux;

		}
		else if (primitiveType == PLANE){
			nX = plane[4 * primitivePosition];
			nY = plane[4 * primitivePosition + 1];
			nZ = plane[4 * primitivePosition + 2];
		}



		for (int l = 0; l<NLIGHTS; l++){

			//calculate light vector
			lX = light[4 * l] - intersectionPointX;
			lY = light[4 * l + 1] - intersectionPointY;
			lZ = light[4 * l + 2] - intersectionPointZ;
			//Normalize l
			aux = 1.0f / sqrtf(lX*lX + lY*lY + lZ*lZ);
			lX *= aux;
			lY *= aux;
			lZ *= aux;

			if (primitiveType == SPHERE){
				//Calculate diffuse shading
				if (sphereLightProperties[primitivePosition * 2]>0){
					aux = lX*nX + lY*nY + lZ*nZ; //dot(n,l)
					if (aux>0){
						aux *= sphereLightProperties[primitivePosition * 2];
						r += aux*sphereColor[primitivePosition * 4] * lightColor[l * 3];
						g += aux*sphereColor[primitivePosition * 4 + 1] * lightColor[l * 3 + 1];
						b += aux*sphereColor[primitivePosition * 4 + 2] * lightColor[l * 3 + 2];
					}
					////Calculate specular component
					if (1.0f - sphereLightProperties[primitivePosition * 2]>0){
						//r=l-2dot(l,n)n
						rX = lX - 2.0f*(lX*nX + lY*nY + lZ*nZ)*nX;
						rY = lY - 2.0f*(lX*nX + lY*nY + lZ*nZ)*nY;
						rZ = lZ - 2.0f*(lX*nX + lY*nY + lZ*nZ)*nZ;

						aux = vX*rX + vY*rY + vZ*rZ;//dot(v,r)
						if (aux>0){
							aux *= powf(aux, 20)*(1.0f - sphereLightProperties[primitivePosition * 2]);
							r += aux*lightColor[l * 3];
							g += aux*lightColor[l * 3 + 1];
							b += aux*lightColor[l * 3 + 2];


						}
					}

				}

			}
			else if (primitiveType == PLANE){
				//Calculate diffuse shading
				if (planeLightProperties[primitivePosition * 2]>0){

					aux = lX*nX + lY*nY + lZ*nZ; //dot(l,n)
					if (aux>0){
						aux *= planeLightProperties[primitivePosition * 2];
						r += aux*planeColor[primitivePosition * 3] * lightColor[l * 3];
						g += aux*planeColor[primitivePosition * 3 + 1] * lightColor[l * 3 + 1];
						b += aux*planeColor[primitivePosition * 3 + 2] * lightColor[l * 3 + 2];
					}
				}

				////Calculate specular component
				if (1.0f - planeLightProperties[primitivePosition * 2]>0){
					//r=l-2dot(l,n)n
					rX = lX - 2.0f*(lX*nX + lY*nY + lZ*nZ)*nX;
					rY = lY - 2.0f*(lX*nX + lY*nY + lZ*nZ)*nY;
					rZ = lZ - 2.0f*(lX*nX + lY*nY + lZ*nZ)*nZ;

					aux = vX*rX + vY*rY + vZ*rZ;//dot(v,r)
					if (aux>0){
						aux *= powf(aux, 20)*(1.0f - planeLightProperties[primitivePosition * 2]);
						r += aux*lightColor[l * 3];
						g += aux*lightColor[l * 3 + 1];
						b += aux*lightColor[l * 3 + 2];


					}
				}

			}

			////Calculate shade
			shade = 0.5f;

			intersections(intersectionPointX + EPSILON*lX, intersectionPointY + EPSILON*lY, intersectionPointZ + EPSILON*lZ, lX, lY, lZ, &primitiveTypeR, &primitivePositionR, &distanceR);
			if (primitiveTypeR == LIGHT){
				shade = 1.0f;
			}

			r *= shade;
			g *= shade;
			b *= shade;

		}

	}

}


__device__ void newRay(float originX, float originY, float originZ, float vX, float vY, float vZ, float &r, float &g, float &b){
	float intersectionPointX, intersectionPointY, intersectionPointZ;
	float nX, nY, nZ, lX, lY, lZ;
	float rX, rY, rZ;
	float distance;
	float shade = 0.0f;
	float aux;
	char primitiveType;
	int primitivePosition;

	float distanceR;
	char primitiveTypeR;
	int primitivePositionR;
	
	//Normalize V
	aux = 1 / sqrtf(vX*vX + vY*vY + vZ*vZ);
	vX *= aux;
	vY *= aux;
	vZ *= aux;

	intersections(originX, originY, originZ, vX, vY, vZ, &primitiveType, &primitivePosition, &distance);

	r = 0.0f;
	g = 0.0f;
	b = 0.0f;
	if (primitiveType == LIGHT){ //if it hit a light, assign the color and stop this ray
		r += lightColor[3 * primitivePosition];
		g += lightColor[3 * primitivePosition + 1];
		b += lightColor[3 * primitivePosition + 2];
		//	isActive=false;
	}
	else if (primitiveType != -1 && distance>0){

		intersectionPointX = originX + distance*vX;
		intersectionPointY = originY + distance*vY;
		intersectionPointZ = originZ + distance*vZ;

		if (primitiveType == SPHERE){
			nX = intersectionPointX - sphere[4 * primitivePosition];
			nY = intersectionPointY - sphere[4 * primitivePosition + 1];
			nZ = intersectionPointZ - sphere[4 * primitivePosition + 2];
			//Normalize n
			aux = 1.0f / sqrtf(nX*nX + nY*nY + nZ*nZ);
			nX *= aux;
			nY *= aux;
			nZ *= aux;

		}
		else if (primitiveType == PLANE){
			nX = plane[4 * primitivePosition];
			nY = plane[4 * primitivePosition + 1];
			nZ = plane[4 * primitivePosition + 2];
		}



		for (int l = 0; l<NLIGHTS; l++){

			//calculate light vector
			lX = light[4 * l] - intersectionPointX;
			lY = light[4 * l + 1] - intersectionPointY;
			lZ = light[4 * l + 2] - intersectionPointZ;
			//Normalize l
			aux = 1.0f / sqrtf(lX*lX + lY*lY + lZ*lZ);
			lX *= aux;
			lY *= aux;
			lZ *= aux;

			if (primitiveType == SPHERE){
				//Calculate diffuse shading
				if (sphereLightProperties[primitivePosition * 2]>0){
					aux = lX*nX + lY*nY + lZ*nZ; //dot(n,l)
					if (aux>0){
						aux *= sphereLightProperties[primitivePosition * 2];
						r += aux*sphereColor[primitivePosition * 4] * lightColor[l * 3];
						g += aux*sphereColor[primitivePosition * 4 + 1] * lightColor[l * 3 + 1];
						b += aux*sphereColor[primitivePosition * 4 + 2] * lightColor[l * 3 + 2];
					}
					////Calculate specular component
					if (1.0f - sphereLightProperties[primitivePosition * 2]>0){
						//r=l-2dot(l,n)n
						rX = lX - 2.0f*(lX*nX + lY*nY + lZ*nZ)*nX;
						rY = lY - 2.0f*(lX*nX + lY*nY + lZ*nZ)*nY;
						rZ = lZ - 2.0f*(lX*nX + lY*nY + lZ*nZ)*nZ;

						aux = vX*rX + vY*rY + vZ*rZ;//dot(v,r)
						if (aux>0){
							aux *= powf(aux, 20)*(1.0f - sphereLightProperties[primitivePosition * 2]);
							r += aux*lightColor[l * 3];
							g += aux*lightColor[l * 3 + 1];
							b += aux*lightColor[l * 3 + 2];


						}
					}


					//Calculate reflection
					if (sphereLightProperties[2 * primitivePosition + 1] > 0)
					{
						//r=v-2dot(v,n)n
						rX = vX - 2.0f*(vX*nX + vY*nY + vZ*nZ)*nX;
						rY = vY - 2.0f*(vX*nX + vY*nY + vZ*nZ)*nY;
						rZ = vZ - 2.0f*(vX*nX + vY*nY + vZ*nZ)*nZ;

						aux = sphereLightProperties[2 * primitivePosition + 1];
						float rRef, gRef, bRef;
						newRay2(intersectionPointX + EPSILON*rX, intersectionPointY + EPSILON*rY, intersectionPointZ + EPSILON*rZ, rX, rY, rZ, rRef, gRef, bRef);

						r += aux *rRef * sphereColor[primitivePosition * 4];
						g += aux*gRef * sphereColor[primitivePosition * 4 + 1];
						b += aux*bRef * sphereColor[primitivePosition * 4 + 2];
					}

				}

			}
			else if (primitiveType == PLANE){
				//Calculate diffuse shading
				if (planeLightProperties[primitivePosition * 2]>0){

					aux = lX*nX + lY*nY + lZ*nZ; //dot(l,n)
					if (aux>0){
						aux *= planeLightProperties[primitivePosition * 2];
						r += aux*planeColor[primitivePosition * 3] * lightColor[l * 3];
						g += aux*planeColor[primitivePosition * 3 + 1] * lightColor[l * 3 + 1];
						b += aux*planeColor[primitivePosition * 3 + 2] * lightColor[l * 3 + 2];
					}
				}

				////Calculate specular component
				if (1.0f - planeLightProperties[primitivePosition * 2]>0){
					//r=l-2dot(l,n)n
					rX = lX - 2.0f*(lX*nX + lY*nY + lZ*nZ)*nX;
					rY = lY - 2.0f*(lX*nX + lY*nY + lZ*nZ)*nY;
					rZ = lZ - 2.0f*(lX*nX + lY*nY + lZ*nZ)*nZ;

					aux = vX*rX + vY*rY + vZ*rZ;//dot(v,r)
					if (aux>0){
						aux *= powf(aux, 20)*(1.0f - planeLightProperties[primitivePosition * 2]);
						r += aux*lightColor[l * 3];
						g += aux*lightColor[l * 3 + 1];
						b += aux*lightColor[l * 3 + 2];


					}
				}


				//Calculate reflection
				if (planeLightProperties[2 * primitivePosition + 1] > 0)
				{
					//r=v-2dot(v,n)n
					rX = vX - 2.0f*(vX*nX + vY*nY + vZ*nZ)*nX;
					rY = vY - 2.0f*(vX*nX + vY*nY + vZ*nZ)*nY;
					rZ = vZ - 2.0f*(vX*nX + vY*nY + vZ*nZ)*nZ;

					aux = planeLightProperties[2 * primitivePosition + 1];
					float rRef, gRef, bRef;
					newRay2(intersectionPointX + EPSILON*rX, intersectionPointY + EPSILON*rY, intersectionPointZ + EPSILON*rZ, rX, rY, rZ, rRef, gRef, bRef);

					r += aux *rRef * planeColor[primitivePosition * 4];
					g += aux*gRef * planeColor[primitivePosition * 4 + 1];
					b += aux*bRef * planeColor[primitivePosition * 4 + 2];
				}

			}

			////Calculate shade
			shade = 0.5f;

			intersections(intersectionPointX + EPSILON*lX, intersectionPointY + EPSILON*lY, intersectionPointZ + EPSILON*lZ, lX, lY, lZ, &primitiveTypeR, &primitivePositionR, &distanceR);
			if (primitiveTypeR == LIGHT){
				shade = 1.0f;
			}

			r *= shade;
			g *= shade;
			b *= shade;



		}

	}


}

__global__ void rayTrace(float originX, float originY, float originZ, float xInf, float yInf, int width, int height, uchar4* output){

	float destX, destY, destZ;     //represents a pixel of the screen on a virtual plane
	float vX, vY, vZ; 		     //vector parallel to the traced ray
	float r = 0.0f, g = 0.0f, b = 0.0f;  //final color
	float intersectionPointX, intersectionPointY, intersectionPointZ;
	float nX, nY, nZ, lX, lY, lZ;
	float rX, rY, rZ;
	float distance;
	float shade=0.0f;
	float aux;
	char primitiveType;
	int primitivePosition;

	float distanceR;
	char primitiveTypeR;
	int primitivePositionR;

	//bool isActive;
	for (int globalTidY = threadIdx.y + blockDim.y*blockIdx.y; globalTidY<height; globalTidY += gridDim.y*blockDim.y){ //stride, in case threads
		for (int globalTidX = threadIdx.x + blockDim.x*blockIdx.x; globalTidX<width; globalTidX += gridDim.x*blockDim.x){//need to do more work

			//Initialize the world coordinate of the point of the screen that each thread will process.
			destX = xInf +globalTidX * 0.01f;
			destY = yInf + globalTidY * 0.01f;
			destZ = 0.0f;

			//We would transform the origin and dest vectors here using the transformation matrix. It's static for now.

			//initialize V that describes the line L(t)=origin+tV. Where origin is a vector that points to the center of the camera,
			//and V a vector parallel to the traced ray.
			vX = destX - originX;
			vY = destY - originY;
			vZ = destZ - originZ;

			//Normalize V
			aux = 1/sqrtf(vX*vX + vY*vY + vZ*vZ);
			vX *= aux;
			vY *= aux;
			vZ *= aux;
		
			intersections(originX, originY, originZ, vX, vY, vZ, &primitiveType, &primitivePosition, &distance);
			
			r = 0.0f;
			g = 0.0f;
			b = 0.0f;
			if (primitiveType == LIGHT){ //if it hit a light, assign the color and stop this ray
				r += lightColor[3 * primitivePosition];
				g += lightColor[3 * primitivePosition + 1];
				b += lightColor[3 * primitivePosition + 2];
				//	isActive=false;
			}
			else if (primitiveType != -1 && distance>0){

				intersectionPointX = originX + distance*vX;
				intersectionPointY = originY + distance*vY;
				intersectionPointZ = originZ + distance*vZ;

				if (primitiveType == SPHERE){
					nX = intersectionPointX - sphere[4 * primitivePosition];
					nY = intersectionPointY - sphere[4 * primitivePosition + 1];
					nZ = intersectionPointZ - sphere[4 * primitivePosition + 2];
					//Normalize n
					aux = 1.0f / sqrtf(nX*nX + nY*nY + nZ*nZ);
					nX *= aux;
					nY *= aux;
					nZ *= aux;

				}
				else if (primitiveType == PLANE){
					nX = plane[4 * primitivePosition];
					nY = plane[4 * primitivePosition + 1];
					nZ = plane[4 * primitivePosition + 2];
				}

				

				for (int l = 0; l<NLIGHTS; l++){
					//calculate light vector
					lX = light[4 * l] - intersectionPointX;
					lY = light[4 * l + 1] - intersectionPointY;
					lZ = light[4 * l + 2] - intersectionPointZ;
					//Normalize l
					aux = 1.0f/sqrtf(lX*lX + lY*lY + lZ*lZ);
					lX *=  aux;
					lY *=  aux;
					lZ *=  aux;

					/*if (globalTidX == 69 && globalTidY == 1){
						printf("light %d %f\n", l, distance);
						printf("ip(%f, %f, %f)\n", intersectionPointX, intersectionPointY, intersectionPointZ);
						printf("l(%f, %f, %f)\n", lX, lY, lZ);
					}*/
					if (primitiveType == SPHERE){
						//Calculate diffuse shading
						if (sphereLightProperties[primitivePosition * 2]>0){
							aux = lX*nX + lY*nY + lZ*nZ; //dot(n,l)
							if (aux>0){
								aux *= sphereLightProperties[primitivePosition * 2];
								r += aux*sphereColor[primitivePosition * 4] * lightColor[l * 3];
								g += aux*sphereColor[primitivePosition * 4 + 1] * lightColor[l * 3 + 1];
								b += aux*sphereColor[primitivePosition * 4 + 2] * lightColor[l * 3 + 2];
							}
							////Calculate specular component
							if (1.0f - sphereLightProperties[primitivePosition * 2]>0){
								//r=l-2dot(l,n)n
								rX = lX - 2.0f*(lX*nX + lY*nY + lZ*nZ)*nX;
								rY = lY - 2.0f*(lX*nX + lY*nY + lZ*nZ)*nY;
								rZ = lZ - 2.0f*(lX*nX + lY*nY + lZ*nZ)*nZ;

								aux = vX*rX + vY*rY + vZ*rZ;//dot(v,r)
								if (aux>0){
									aux *= powf(aux, 20)*(1.0f - sphereLightProperties[primitivePosition * 2]);
									r += aux*lightColor[l * 3];
									g += aux*lightColor[l * 3 + 1];
									b += aux*lightColor[l * 3 + 2];

									
								}
							}
								

							//Calculate reflection
							if (sphereLightProperties[2*primitivePosition+1] > 0)
							{
								//r=v-2dot(v,n)n
								rX = vX - 2.0f*(vX*nX + vY*nY + vZ*nZ)*nX;
								rY = vY - 2.0f*(vX*nX + vY*nY + vZ*nZ)*nY;
								rZ = vZ - 2.0f*(vX*nX + vY*nY + vZ*nZ)*nZ;
							
								aux = sphereLightProperties[2 * primitivePosition + 1];
								float rRef, gRef,bRef;
								newRay(intersectionPointX + EPSILON*rX, intersectionPointY + EPSILON*rY, intersectionPointZ + EPSILON*rZ, rX, rY, rZ, rRef, gRef, bRef);
								
								r+= aux *rRef * sphereColor[primitivePosition*4];
								g +=aux*gRef * sphereColor[primitivePosition * 4 + 1];
								b +=aux*bRef * sphereColor[primitivePosition * 4 + 2];
							}

						}

					}
					else if (primitiveType == PLANE){
						//Calculate diffuse shading
						if (planeLightProperties[primitivePosition * 2]>0){

							aux = lX*nX + lY*nY + lZ*nZ; //dot(l,n)
							if (aux>0){
								aux *= planeLightProperties[primitivePosition * 2];
								r += aux*planeColor[primitivePosition * 3] * lightColor[l * 3];
								g += aux*planeColor[primitivePosition * 3 + 1] * lightColor[l * 3 + 1];
								b += aux*planeColor[primitivePosition * 3 + 2] * lightColor[l * 3 + 2];
							}
						}

						////Calculate specular component
						if (1.0f - planeLightProperties[primitivePosition * 2]>0){
							//r=l-2dot(l,n)n
							rX = lX - 2.0f*(lX*nX + lY*nY + lZ*nZ)*nX;
							rY = lY - 2.0f*(lX*nX + lY*nY + lZ*nZ)*nY;
							rZ = lZ - 2.0f*(lX*nX + lY*nY + lZ*nZ)*nZ;

							aux = vX*rX + vY*rY + vZ*rZ;//dot(v,r)
							if (aux>0){
								aux *= powf(aux, 20)*(1.0f - planeLightProperties[primitivePosition * 2]);
								r += aux*lightColor[l * 3];
								g += aux*lightColor[l * 3 + 1];
								b += aux*lightColor[l * 3 + 2];


							}
						}


						//Calculate reflection
						if (planeLightProperties[2 * primitivePosition + 1] > 0)
						{
							//r=v-2dot(v,n)n
							rX = vX - 2.0f*(vX*nX + vY*nY + vZ*nZ)*nX;
							rY = vY - 2.0f*(vX*nX + vY*nY + vZ*nZ)*nY;
							rZ = vZ - 2.0f*(vX*nX + vY*nY + vZ*nZ)*nZ;

							aux = planeLightProperties[2 * primitivePosition + 1];
							float rRef, gRef, bRef;
							newRay(intersectionPointX + EPSILON*rX, intersectionPointY + EPSILON*rY, intersectionPointZ + EPSILON*rZ, rX, rY, rZ, rRef, gRef, bRef);

							r += aux *rRef * planeColor[primitivePosition * 4];
							g += aux*gRef * planeColor[primitivePosition * 4 + 1];
							b += aux*bRef * planeColor[primitivePosition * 4 + 2];
						}

					}

					////Calculate shade
					shade = 0.5f;

					intersections(intersectionPointX + EPSILON*lX, intersectionPointY + EPSILON*lY, intersectionPointZ + EPSILON*lZ, lX, lY, lZ, &primitiveTypeR, &primitivePositionR, &distanceR);
					if (primitiveTypeR == LIGHT){
						shade = 1.0f;
					}

					r *= shade; 
					g *= shade;
					b *= shade;		



				}

			}
			else{
				//	isActive=false;
			}
		
			output[globalTidY*width + globalTidX].x = (int)min(256.0f*r, 255.0f);
			output[globalTidY*width + globalTidX].y = (int)min(256.0f*g, 255.0f);
			output[globalTidY*width + globalTidX].z = (int)min(256.0f*b, 255.0f);
			output[globalTidY*width + globalTidX].w = 255;


		}
	}
}

//Scene initialization
/*float sphereH[NSPHERES * 4] = {0.0f, -1.0f, 10.0f, 1.5f,
-3.5f, -1.0f, 10.0f, 1.5f,
3.5f, -1.0f, 10.0f, 1.5f,
0.5f, 0.0f, 8.0f, 0.5f };
float sphereColorH[NSPHERES * 4] = { 1.0f, 0.0f, 0.0f, 0.0f,
0.0f, 1.0f, 0.0f, 0.0f,
0.0f, 0.0f, 1.0f, 0.0f,
1.0f, 0.5f, 0.0f, 0.0f };
float sphereLightPropertiesH[NSPHERES * 2] = {0.6f, 0.1f,
0.9f, 0.1f,
0.1f, 1.0f,
0.8f, 0.1f };

float planeH[NPLANES * 4] = { 0.0f, 1.0f, 0.0f, 4.0f };
float planeColorH[NPLANES * 3] = { 0.65f, 0.65f, 0.65f };
float planeLightPropertiesH[NPLANES * 2] = { 0.5f, 0.5f };

float lightH[NLIGHTS * 4] = { 3.0f, 1.0f, 0.0f, 0.2f };
float lightColorH[NLIGHTS * 3] = { 0.95f, 0.95f, 0.95f };
*/

float sphereH[NSPHERES * 4] = { -5.5f, -2.4, 7.0f, 2.0f,
								0.0f, -2.4, 7.0f, 2.0f, 
								 5.5f, -2.4, 7.0f, 2.0f };
float sphereColorH[NSPHERES * 4] = { 0.9f, 0.2f, 0.2f, 0.0,
									0.2f, 0.9f, 0.2f, 0.0f,
									0.2f, 0.2f, 0.9f, 0.0f };
float sphereLightPropertiesH[NSPHERES * 2] = { 0.8f, 0.8f,
												0.1f, 1.0f,
												0.2f, 0.8f };

float planeH[NPLANES * 4] = { 0.0f, 1.0f, 0.0f, 4.4f };
float planeColorH[NPLANES * 3] = { 0.4f, 0.4f, 0.4f };
float planeLightPropertiesH[NPLANES * 2] = { 0.5f, 0.8f };

float lightH[NLIGHTS * 4] = { 0.0f, 5.0f, 5.0f, 0.1f,
							2.0f, 5.0f, 1.0f, 0.1f };
float lightColorH[NLIGHTS * 3] = { 0.6f, 0.6f, 0.6f,
									0.9f, 0.9f, 0.9f };





static void draw_func(void) {
	// we pass zero as the last parameter, because out bufferObj is now
	// the source, and the field switches from being a pointer to a
	// bitmap to now mean an offset into a bitmap object
	glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
	{
		lightH[0] += 0.5f;
		HANDLE_ERROR(cudaMemcpyToSymbol(light, lightH, NLIGHTS * 4 * sizeof(float)));
	}
	else if (key == GLFW_KEY_LEFT && action == GLFW_PRESS)
	{
		lightH[0] -= 0.5f;
		HANDLE_ERROR(cudaMemcpyToSymbol(light, lightH, NLIGHTS * 4 * sizeof(float)));
	}
	else if (key == GLFW_KEY_UP && action == GLFW_PRESS)
	{
		lightH[1] += 0.5f;
		HANDLE_ERROR(cudaMemcpyToSymbol(light, lightH, NLIGHTS * 4 * sizeof(float)));
	}
	else if (key == GLFW_KEY_DOWN && action == GLFW_PRESS)
	{
		lightH[1] -= 0.5f;
		HANDLE_ERROR(cudaMemcpyToSymbol(light, lightH, NLIGHTS * 4 * sizeof(float)));
	}
}


int main(int argc, char **argv) {
	

	//copy scene data to constant memory on GPU
	HANDLE_ERROR(cudaMemcpyToSymbol(sphere, sphereH, NSPHERES * 4 * sizeof(float)));
	HANDLE_ERROR(cudaMemcpyToSymbol(sphereColor, sphereColorH, NSPHERES * 4 * sizeof(float)));
	HANDLE_ERROR(cudaMemcpyToSymbol(sphereLightProperties, sphereLightPropertiesH, NSPHERES * 2 * sizeof(float)));
	HANDLE_ERROR(cudaMemcpyToSymbol(plane, planeH, NPLANES * 4 * sizeof(float)));
	HANDLE_ERROR(cudaMemcpyToSymbol(planeColor, planeColorH, NPLANES * 3 * sizeof(float)));
	HANDLE_ERROR(cudaMemcpyToSymbol(planeLightProperties, planeLightPropertiesH, NPLANES * 2 * sizeof(float)));
	HANDLE_ERROR(cudaMemcpyToSymbol(light, lightH, NLIGHTS * 4 * sizeof(float)));
	HANDLE_ERROR(cudaMemcpyToSymbol(lightColor, lightColorH, NLIGHTS * 3 * sizeof(float)));
	


	
	cudaDeviceProp  prop;
	int dev;



	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	HANDLE_ERROR(cudaChooseDevice(&dev, &prop));

	// tell CUDA which dev we will be using for graphic interop
	// from the programming guide:  Interoperability with OpenGL
	//     requires that the CUDA device be specified by
	//     cudaGLSetGLDevice() before any other runtime calls.

	//HANDLE_ERROR(cudaGLSetGLDevice(dev));

	GLFWwindow* window;

	// Initialise GLFW
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}


	// Open a window and create its OpenGL context
	window = glfwCreateWindow(WIDTH, HEIGHT, "bitmap", NULL, NULL);
	if (window == NULL){
		fprintf(stderr, "Failed to open GLFW window.\n");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);

	//glfwSwapInterval(1);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}



	/* Make the window's context current */
	glfwMakeContextCurrent(window);
	//leer teclado 
	glfwSetKeyCallback(window, key_callback);

	// the first three are standard OpenGL, the 4th is the CUDA reg 
	// of the bitmap these calls exist starting in OpenGL 1.5
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, WIDTH * HEIGHT * 4,
		NULL, GL_DYNAMIC_DRAW_ARB);

	HANDLE_ERROR(
		cudaGraphicsGLRegisterBuffer(&resource,
		bufferObj,
		cudaGraphicsMapFlagsNone));
	/* Loop GLFW until the user closes the window */
	while (!glfwWindowShouldClose(window))
	{
		
		// do work with the memory dst being on the GPU, gotten via mapping
		HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, NULL));
		uchar4* devPtr;
		size_t  size;
		HANDLE_ERROR(
			cudaGraphicsResourceGetMappedPointer((void**)&devPtr,
			&size,
			resource));

		rayTrace <<<dim3(25, 20), dim3(32, 32) >>>(0, 0, -5, -4.0f, -3.0f, WIDTH, HEIGHT, devPtr);
		/* Render here */
		
		HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));
		draw_func();
		

		/* Swap front and back buffers */
		glfwSwapBuffers(window);

		/* Poll for and process events */
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;

}
