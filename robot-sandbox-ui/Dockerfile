# --- STAGE 1: Build the React Application ---
FROM node:lts-alpine AS build-stage

WORKDIR /app

# Copy package configurations and install dependencies
COPY package*.json ./
RUN npm install

# Copy all source files and build the production distribution
COPY . .
RUN npm run build

# --- STAGE 2: Serve via Nginx (Production-Grade) ---
FROM nginx:stable-alpine AS production-stage

# Copy the compiled production assets from the build stage to Nginx html directory
# (If using Vite, output is 'dist'. If using Create React App, output is 'build')
COPY --from=build-stage /app/dist /usr/share/nginx/html

# Expose port 80 inside the container
EXPOSE 80

# Start Nginx server
CMD ["nginx", "-g", "daemon off;"]