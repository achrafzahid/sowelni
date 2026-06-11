# darija-transcriber-frontend/Dockerfile

# --- Stage 1: Build the Angular Application ---
FROM node:18-alpine AS build
WORKDIR /app

# Copy package management files and install dependencies
COPY package*.json ./
RUN npm ci

# Copy the rest of the frontend source code
COPY . .

# Build the app in production mode
RUN npm run build -- --configuration=production

# --- Stage 2: Serve with Nginx ---
FROM nginx:alpine

# Copy the compiled static production build from Stage 1 into Nginx default directory
# Note: Adjust 'darija-transcriber-frontend' if your angular.json build output folder name differs
COPY --from=build /app/dist/darija-transcriber-frontend /usr/share/nginx/html

# Expose port 80 for regular browser access
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]