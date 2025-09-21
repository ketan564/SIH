import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Droplets, Wind, Gauge, Eye } from "lucide-react";

interface WeatherCondition {
  text: string;
  icon: string;
}

interface WeatherCurrent {
  temp_c: number;
  humidity: number;
  wind_kph: number;
  wind_dir: string;
  pressure_mb: number;
  vis_km: number;
  uv: number;
  condition: WeatherCondition;
}

interface WeatherLocation {
  name: string;
  region: string;
}

interface WeatherApiResponse {
  location: WeatherLocation;
  current: WeatherCurrent;
}

interface WeatherCardProps {
  onLocation?: (loc: string) => void;
}

const WeatherCard: React.FC<WeatherCardProps> = ({ onLocation }) => {
  const [weather, setWeather] = useState<WeatherApiResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const apiKey = process.env.NEXT_PUBLIC_WEATHER_API_KEY;

  useEffect(() => {
    if (!navigator.geolocation) {
      setError("Geolocation is not supported by your browser.");
      setLoading(false);
      return;
    }

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        try {
          const lat = position.coords.latitude;
          const lon = position.coords.longitude;

          const res = await fetch(
            `https://api.weatherapi.com/v1/current.json?key=${apiKey}&q=${lat},${lon}`
          );

          if (!res.ok) throw new Error("Failed to fetch weather data.");

          const data: WeatherApiResponse = await res.json();
          setWeather(data);

          // Pass location string up to parent if onLocation prop provided
          if (data.location && onLocation) {
            onLocation(`${data.location.name}, ${data.location.region}`);
          }
        } catch (err) {
          setError("Failed to load weather data.");
        } finally {
          setLoading(false);
        }
      },
      () => {
        setError("Location permission denied.");
        setLoading(false);
      }
    );
  }, [apiKey, onLocation]);

  if (loading) return <div>Loading weather data...</div>;
  if (error) return <div className="text-destructive">Error: {error}</div>;
  if (!weather) return <div>No weather data available.</div>;

  function getUVLevel(uv: number) {
    if (uv <= 2) return { level: "Low", color: "text-green-600 bg-green-100" };
    if (uv <= 5) return { level: "Moderate", color: "text-yellow-600 bg-yellow-100" };
    if (uv <= 7) return { level: "High", color: "text-orange-600 bg-orange-100" };
    if (uv <= 10) return { level: "Very High", color: "text-red-600 bg-red-100" };
    return { level: "Extreme", color: "text-purple-600 bg-purple-100" };
  }

  const uvLevel = getUVLevel(weather.current.uv);

  return (
    <div className="max-w-md mx-auto p-4 bg-card rounded-lg shadow-md">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {weather.current.condition.icon && (
              <img
                src={`https:${weather.current.condition.icon}`}
                alt={weather.current.condition.text}
                width={32}
                height={32}
                className="inline-block"
              />
            )}
            <span className="font-semibold text-lg">
              Weather in {weather.location.name}, {weather.location.region}
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <p className="text-xl font-bold flex items-center gap-2">
              🌡️ {weather.current.temp_c} °C
            </p>
            <p className="flex items-center gap-2 text-lg">
              <Droplets className="h-5 w-5 text-primary" />
              Humidity: {weather.current.humidity}%
            </p>
            <p className="flex items-center gap-2 text-lg">
              <Wind className="h-5 w-5 text-primary" />
              Wind: {weather.current.wind_kph} km/h {weather.current.wind_dir}
            </p>
            <p className="flex items-center gap-2 text-lg">
              <Gauge className="h-5 w-5 text-primary" />
              Pressure: {weather.current.pressure_mb} hPa
            </p>
            <p className="flex items-center gap-2 text-lg">
              <Eye className="h-5 w-5 text-primary" />
              Visibility: {weather.current.vis_km} km
            </p>
          </div>
          <div className="mt-4 flex items-center gap-2">
            <span className="text-sm text-muted-foreground">UV Index:</span>
            <Badge className={uvLevel.color}>
              {weather.current.uv} - {uvLevel.level}
            </Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default WeatherCard;
